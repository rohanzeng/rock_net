# Calculate various score metrics relative to a trained model and evaluate

#----------------------
# USER SPECIFIED DATA
#----------------------

import numpy as np
import pandas as pd
import os
import natsort
import copy
import time
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets, utils, models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
#from skimage import io
#import helper

import NET_FCN as nf

num_classes = 2
image_shape = (332, 514)
num_epochs = 100
batch_size = 1
a = 1e-4
weight_decay = 1e-4
use_gpu = True

data_dir = "../data"
runs_dir = "./outputs"
base_dir = "../data/all_navcam/output"
label_dir = "../data/all_navcam/outputL"
vgg_path = "../data/rvgg"

modelTest = "57216.torch"
setType = "valid" #"valid" or "test"
thresh = 0.15

UseValidationSet = False
TrainedModelWeightDir = "TrainedModelWeights/"
Trained_model_path = TrainedModelWeightDir+modelTest
TrainLossTxtFile = TrainedModelWeightDir + "TrainLoss.txt"
ValidLossTxtFile = TrainedModelWeightDir + "ValidLoss.txt"
Pretrained_Encoder_Weights = 'densenet_cosine_264_k32.pth'
ScoreTxtFile = runs_dir + "/scoreFile.txt"

#Dataset Class: Gets items and returns the index for access by other datasets
class RockDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, idx #, self.total_imgs[idx]

def load_data():
    #Define transforms
    #resize = transforms.Resize(256)
    #crop = transforms.CenterCrop(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    t_transform = transforms.Compose([transforms.ToTensor(), normalize])
    l_transform = transforms.Compose([transforms.ToTensor()])

    #Create labeled [l] and train [t] dataset classes, split into train and validation sets
    t_set = RockDataSet(base_dir + '/train/left_jpgs', t_transform)
    v_set = RockDataSet(base_dir + '/val/left_jpgs', t_transform)
    te_set = RockDataSet(base_dir + '/test/left_jpgs', t_transform)
    lt_set = RockDataSet(label_dir + '/train/labels_pngs', l_transform)
    lv_set = RockDataSet(label_dir + '/val/labels_pngs', l_transform)
    lte_set = RockDataSet(label_dir + '/test/labels_pngs', l_transform)

    #Set up loader for both train and validation sets
    t_loader = DataLoader(t_set, batch_size = batch_size, shuffle = True)
    v_loader = DataLoader(v_set, batch_size = batch_size, shuffle = False)
    te_loader = DataLoader(te_set, batch_size = batch_size, shuffle = True)

    datasets = {'train_label': lt_set, 'valid_label': lv_set, 'test_label': lte_set, 'train': t_set, 'valid': v_set, 'test': te_set}
    loaders = {'train': t_loader, 'valid': v_loader, 'test': te_loader}


    return datasets, loaders

def load_model():

    model = nf.Net(NumClasses = num_classes, PreTrainedModelPath = Pretrained_Encoder_Weights, UseGPU = use_gpu,
            UpdateEncoderBatchNormStatistics = True)

    model.load_state_dict(torch.load(Trained_model_path))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model

def decode_segmap(image, bc, nc = num_classes):
    label_colors = np.array([(128,0,0),(0,128,0)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l,0]
        g[idx] = label_colors[l,1]
        b[idx] = label_colors[l,2]

    count = 0
    rgb = np.stack([r, g, b], axis = 2)

    return rgb

def reverse_segmap(image):
    fin = np.zeros_like(image[0]).astype(np.uint8)
    base = image[1] == 128/255
    fin[base] = 1
    return fin

def compute_bbox_coordinates(mask_img, probs, lookup_range, verbose = 0):

    bbox_list = list()


    img = Image.fromarray(np.uint8(mask_img*255), 'L')
    thresh = cv2.threshold(np.array(img),128,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        count = 0
        score = 0
        bbox = {'i_min': y, 'j_min': x, 'i_max': y+h, 'j_max': x+w, 'score': 0}
        bbox_list.append(bbox)
    return bbox_list

def run():

    ###Load Data
    datasets, loaders = load_data()

    #Test to save image
    '''
    im, lab = next(iter(t_loader))
    plt.imshow(im[0,:,:,:].permute(1,2,0))
    plt.savefig('im1.png')
    plt.imshow(l_dataset[lab][0].permute(1,2,0))
    plt.savefig('im1l.png')
    #'''
    #print(len(l_dataset)) 10310

    print("Train Data Loaded")

    ### Load Model and Parameters
    model = load_model()

    print('Model Data Loaded')

    # Sam's outputs are the labeled data, Rohan's outputs are the outputs from the network
    
    # Number of rocks each network has found
    samBox = 0
    rohBox = 0

    # Average number of rocks
    avgRSBox = 0

    # Average IoU and Dice Score
    avgIoU = 0
    avgDice = 0

    # Average Adjusted IoU and Dice Score (not including False Positives since the network may label some pixels correctly that were incorrectly labeled in the training labels
    avgAdjIoU = 0
    avgAdjDice = 0

    # Average IoU and Dice Score around the bounding boxes in Sam's outputs
    avgSIoU = 0
    avgSDice = 0

    # Average positive pixels (rock pixels)
    avgPos = 0

    #(c0, c1) = (0.7, 0.3)
    #avgFalse = 0
    #avgSFalse = 0

    setLen = len(datasets[setType])

    #Visualizing
    setItr = iter(loaders[setType])
    for itr in range(setLen):
        im, ind = next(setItr)
        im = im.permute(0,2,3,1)
        out = model(im)[0]
        prob = out.detach().cpu().numpy()[0,:,:,:] #Probability matrix
        #print(a[0,:,:])
        #print(a[1,:,:])
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        om = (out.squeeze())[1].detach().cpu().numpy()
        #print(om)
        #print(om.shape)
        #print(omi)
        #print(omi.shape)

        rohBase = (om >= thresh) #.astype(int) #0's as background, 1's as rocks
        #print(rohBase)
        #print(rohBase.shape)
        #print(om == 1)
        #print((om == 1).shape)
        roh_bbox = compute_bbox_coordinates(rohBase, prob, lookup_range = 5, verbose = 0) #bbox

        samBase = reverse_segmap(datasets[setType+'_label'][ind][0]) #0's as background, 1's as rocks
        sam_bbox = compute_bbox_coordinates(samBase, prob, lookup_range = 5, verbose = 0) #bbox

        rohBox += len(roh_bbox)
        samBox += len(sam_bbox)

        if len(sam_bbox) != 0:
            avgRSBox += len(roh_bbox)/len(sam_bbox)

        TP = (rohBase&samBase).sum()
        TN = ((1-rohBase)&(1-samBase)).sum()
        FP = ((samBase^rohBase)&rohBase).sum()
        FN = ((samBase^rohBase)&(1-rohBase)).sum()

        if (TP+FP+FN != 0):
            IoU = TP/(TP+FP+FN)
        if (TP+TP+FP+FN != 0):
            Dice = (TP+TP)/(TP+TP+FP+FN)

        if (TP+FN != 0):
            adjIoU = TP/(TP+FN)
        if (TP+TP+FN != 0):
            adjDice = (TP+TP)/(TP+TP+FN)

        avgIoU += IoU
        avgDice += Dice

        avgAdjIoU += adjIoU
        avgAdjDice += adjDice

        sIoU = 0
        sDice = 0
        #sFalse = 0

        if len(sam_bbox) != 0:
            for box in sam_bbox:
                sBox = samBase[box['i_min']:box['i_max'], box['j_min']:box['j_max']]
                rBox = rohBase[box['i_min']:box['i_max'], box['j_min']:box['j_max']]
                miniTP = (rBox&sBox).sum()
                miniTN = ((1-rBox)&(1-sBox)).sum()
                miniFP = ((sBox^rBox)&rBox).sum()
                miniFN = ((sBox^rBox)&(1-rBox)).sum()
                sIoU +=  miniTP/(miniTP+miniFP+miniFN)
                sDice += (miniTP+miniTP)/(miniTP+miniTP+miniFP+miniFN)
                #sFalse += 1-((c0*miniFN+c1*miniFP)/(c0*miniFN+c1*miniFP+miniTN+miniTP))

            avgSIoU += sIoU/len(sam_bbox)
            avgSDice += sDice/len(sam_bbox)

        if samBase.sum() != 0:
            pos = rohBase.sum()/samBase.sum()
            avgPos += pos

        #avgFalse += 1-((c0*FN+c1*FP)/(c0*FN+c1*FP+TN+TP))
        #avgSFalse += sFalse/len(sam_bbox)


        print('Image ' + str(itr+1) + '/' + str(setLen) + ' tested')

    avgRSBox /= setLen
    avgIoU /= setLen
    avgDice /= setLen
    avgAdjIoU /= setLen
    avgAdjDice /= setLen
    avgSIoU /= setLen
    avgSDice /= setLen
    avgPos /= setLen
    #avgFalse /= setLen
    #avgSFalse /= setLen

    with open(ScoreTxtFile, "a") as f:
        f.write("Model "+str(modelTest)+" results on "+setType+" set, thresh = "+str(thresh)+"\n"+
        "Total Number of Rock Clusters:"+
        "(Sam, Rohan) = ("+str(samBox)+", "+str(rohBox)+")\n"+"Average Number of Rock Clusters: (Sam, Rohan)"+
        "= ("+str(samBox/setLen)+", "+str(rohBox/setLen)+")\n"+"Average Ratio Rohan:Sam Boxes = "+
        str(avgRSBox)+"\n"+"Average IoU and Dice Scores: (IoU, Dice) = ("+str(avgIoU)+", "+str(avgDice)+
        ")\n"+"Average Adjusted IoU and Dice Scores: (IoU, Dice) = ("+str(avgAdjIoU)+", "+str(avgAdjDice)+
        ")\n"+"Average IoU and Dice Scores over Sam's Boxes: (IoU, Dice) = ("+str(avgSIoU)+", "+
        str(avgSDice)+")\n"+"Average Ratio Rohan:Sam Rock Pixels = "+str(avgPos)+"\n\n")
        f.close()
        #"Average False/Total Score = "+
        #str(avgFalse)+"\n"+"Average False/Total Score over Sam's Boxes = "+str(avgSFalse)+"\n\n")
        #f.close()

    print("Model "+str(modelTest)+" results on "+setType+" set")
    print("Total Number of Rock Clusters: (Sam, Rohan) = ("+str(samBox)+", "+str(rohBox)+")")
    print("Average Number of Rock Clusters: (Sam, Rohan) = ("+str(samBox/setLen)+", "+str(rohBox/setLen)+")")
    print("Average Ratio Rohan:Sam Boxes = "+str(avgRSBox))
    print("Average IoU and Dice Scores: (IoU, Dice) = ("+str(avgIoU)+", "+str(avgDice)+")")
    print("Average Adjusted IoU and Dice Scores: (IoU, Dice) = ("+str(avgAdjIoU)+", "+str(avgAdjDice)+")")
    print("Average IoU and Dice Scores over Sam's Boxes: (IoU, Dice) = ("+str(avgSIoU)+", "+str(avgSDice)+")")
    print("Average Ratio Rohan:Sam Rock Pixels = "+str(avgPos))
    #print("Average False/Total Score = "+str(avgFalse))
    #print("Average False/Total Score over Sam's Boxes = "+str(avgSFalse))
    return


#----------------------
# MAIN
#----------------------

if __name__ == "__main__":
    run()
