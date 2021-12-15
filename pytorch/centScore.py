# Scores a model based on the centroids of the predicted rocks

#----------------------
# USER SPECIFIED DATA
#----------------------
import numpy as np
#import pandas as pd
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

#import cv2

import scipy as sp
from scipy import optimize

'''
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
from sklearn.metrics import jaccard_similarity_score as jsc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
#from skimage import io
#import helper

import preprocess as pp
import Net_Fcn_Mod as nf
'''

num_classes = 2
image_shape = (332, 514)
num_epochs = 100
batch_size = 1
a = 1e-4
weight_decay = 1e-4
use_gpu = False

data_dir = "../data"
runs_dir = "./outputs"
base_dir = "../data/all_navcam/output"
label_dir = "../data/all_navcam/outputL"
vgg_path = "../data/rvgg"

modelTest = "7002.torch"
setType = "valid" #"valid" or "test"
thresh = 0.15

UseValidationSet = False
TrainedModelWeightDir = "TrainedModelWeights/"
Trained_model_path = TrainedModelWeightDir+modelTest
TrainLossTxtFile = TrainedModelWeightDir + "TrainLoss.txt"
ValidLossTxtFile = TrainedModelWeightDir + "ValidLoss.txt"
Pretrained_Encoder_Weights = 'densenet_cosine_264_k32.pth'
ScoreTxtFile = runs_dir + "/centScoreFile.txt"

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

    #model.load_state_dict(torch.load(Trained_model_path), map_location=torch.device('cpu'))
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

def compute_centroids(bbox_list):
    centList = []
    for bbox in bbox_list:
        yCen = int((bbox['i_min']+bbox['i_max'])/2)
        xCen = int((bbox['j_min']+bbox['j_max'])/2)
        centList.append((xCen, yCen))
    return np.array(centList)

# Sam's indices are the rows, mine are the columns
# Creates an H Matrix to optimize in order to match corresponding centroids, based off of the Hungarian Algorithm
def compute_HMatrix(centS, centR):
    size = max(len(centS), len(centR))
    dif = len(centS)-len(centR)
    if len(centS) > 0:
        sX = centS[:,0]
        sY = centS[:,1]
    else:
        sX = np.array([0])
        sY = np.array([0])
        dif += 1
        centS = np.zeros((1,2))
    if len(centR) > 0:
        rX = centR[:,0]
        rY = centR[:,1]
    else:
        rX = np.array([0])
        rY = np.array([0])
        dif -= 1
        centR = np.zeros((1,2))
    if dif > 0:
        app = np.zeros((1,dif))
        rX = np.append(rX, app).astype(int)
        rY = np.append(rY, app).astype(int)
        app2 = np.zeros((dif,2))
        centR = np.append(centR,app2,axis=0)
    if dif < 0:
        app = np.zeros((1,-dif))
        sX = np.append(sX, app).astype(int)
        sY = np.append(sY, app).astype(int)
        app2 = np.zeros((dif,2))
        centS = np.append(centS,app2,axis=0)
    assert(len(sX) == len(rX))
    assert(len(sY) == len(rY))
    sXM = np.repeat(np.array([sX]), size, axis = 0).T
    sYM = np.repeat(np.array([sY]), size, axis = 0).T
    rXM = np.repeat(np.array([rX]), size, axis = 0)
    rYM = np.repeat(np.array([rX]), size, axis = 0)
    HMatrix = np.sqrt(((sXM-rXM)**2)+((sYM-rYM)**2))
    return HMatrix, centS, centR

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

    avgRec = 0
    avgDis = 0
    avgCov = 0

    passDis = 0
    passCov = 0

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

        TP = (rohBase&samBase).sum()
        TN = ((1-rohBase)&(1-samBase)).sum()
        FP = ((samBase^rohBase)&rohBase).sum()
        FN = ((samBase^rohBase)&(1-rohBase)).sum()

        if (TP+FN != 0):
            adjRec = TP/(TP+FN)

        avgRec += adjRec

        #2D list of centroid coordinates
        centSList = compute_centroids(sam_bbox)
        centRList = compute_centroids(roh_bbox)

        #2D list of matched centroid indices (2xnum centroids)
        #This was based off of the Hungarian Algorithm to find the cost matrix HMatrix
        HMatrix, centSList, centRList = compute_HMatrix(centSList, centRList)
        clusCent = sp.optimize.linear_sum_assignment(HMatrix)

        #I thought that clusCent was ((a1,b1),(a2,b2)) when it's ((a1,a2,a3),(b1,b2,b3))
        #Option 1: Centroid Distance (If the matched centroids are within a certain distance of each other, add a point)
        base = len(centSList)
        dif = len(centSList)-len(centRList)
        disThresh = 150
        disScore = 0
        for ind in range(len(clusCent[0])):
            centInd = np.array([clusCent[0][ind], clusCent[1][ind]])
            #print(clusCent)
            #print(centSList)
            #print(centRList)
            #print(centInd)
            if dif > 0:
                #print("Whee1")
                if centInd[1] >= len(centRList):
                    pass
                else:
                    centR = centRList[centInd[1]]
                    centS = centSList[centInd[0]]
                    #print((centR, centS))
                    curDis = np.sqrt(((centS[0]-centR[0])**2)+((centS[1]-centR[1])**2))
                    if curDis <= disThresh:
                        disScore += 1
            elif dif < 0:
                #print("Whee2")
                if centInd[0] >= len(centSList):
                    pass
                else:
                    centR = centRList[centInd[1]]
                    centS = centSList[centInd[0]]
                    #print((centR, centS))
                    curDis = np.sqrt(((centS[0]-centR[0])**2)+((centS[1]-centR[1])**2))
                    if curDis <= disThresh:
                        disScore += 1
            else:
                #print("Whee3")
                centR = centRList[centInd[1]]
                centS = centSList[centInd[0]]
                #print((centR, centS))
                curDis = np.sqrt(((centS[0]-centR[0])**2)+((centS[1]-centR[1])**2))
                if curDis <= disThresh:
                    disScore += 1
            #print(curDis)
        if base > 0:
            disScore = disScore/base
            avgDis += disScore
        else:
            passDis += 1

        #Option 2: Centroid Coverage (If the predicted centroid lies in the space of the rock of the labeled centroid, add a point)
        base = len(centSList)
        dif = len(centSList)-len(centRList)
        covScore = 0
        #for centInd in clusCent:
        for ind in range(len(clusCent[0])):
            centInd = np.array([clusCent[0][ind], clusCent[1][ind]])
            if len(sam_bbox) == 0:
                pass
            elif dif > 0:
                #print("Whoo1")
                if centInd[1] >= len(centRList):
                    pass
                else:
                    centR = centRList[centInd[1]]
                    sBox = sam_bbox[centInd[0]]
                    #print(centR)
                    #print(sBox)
                    if (sBox['i_min'] <= centR[1] <= sBox['i_max']) and (sBox['j_min'] <= centR[0] <= sBox['j_max']):
                        covScore += 1
            elif dif < 0:
                #print("Whoo2")
                if centInd[0] >= len(centSList):
                    pass
                else:
                    centR = centRList[centInd[1]]
                    sBox = sam_bbox[centInd[0]]
                    #print(centR)
                    #print(sBox)
                    if (sBox['i_min'] <= centR[1] <= sBox['i_max']) and (sBox['j_min'] <= centR[0] <= sBox['j_max']):
                        covScore += 1
            else:
                #print("Whoo3")
                centR = centRList[centInd[1]]
                sBox = sam_bbox[centInd[0]]
                #print(centR)
                #print(sBox)
                if (sBox['i_min'] <= centR[1] <= sBox['i_max']) and (sBox['j_min'] <= centR[0] <= sBox['j_max']):
                    covScore += 1
        if base > 0:
            covScore = covScore/base
            avgCov += covScore
        else:
            passCov += 1

        #print('curRec = '+str(adjRec))
        #print('curDis = '+str(disScore))
        #print('curCov = '+str(covScore))
        print('Image ' + str(itr+1) + '/' + str(setLen) + ' tested')

    avgRec /= setLen
    avgDis /= (setLen-passDis)
    avgCov /= (setLen-passCov)

    print("Model "+str(modelTest)+" results on "+setType+" set, thresh = "+str(thresh))
    print("Recall = "+str(avgRec))
    print("Average Centroid Distance Score (Distance = "+str(disThresh)+"):"+str(avgDis))
    print("Average Centroid Coverage Score:"+str(avgCov))

    with open(ScoreTxtFile, "a") as f:
        f.write("Model "+str(modelTest)+" results on "+setType+" set, thresh = "+str(thresh)+"\n"+
        "Recall Score:"+str(avgRec)+"\n"+"Average Centroid Distance Score (Distance = "+str(disThresh)+
        "):"+str(avgDis)+"\n"+"Average Centroid Coverage Score:"+str(avgCov)+"\n\n")
        f.close()
        #"Average False/Total Score = "+
        #str(avgFalse)+"\n"+"Average False/Total Score over Sam's Boxes = "+str(avgSFalse)+"\n\n")
        #f.close()

    #print("Model "+str(modelTest)+" results on "+setType+" set")
    #print("Recall = "+str(avgRec))
    #print("Average Centroid Distance Score (Distance = "+str(disThresh)+"):"+str(avgDis))
    #print("Average Centroid Coverage Score:"+str(avgCov))
    return


#----------------------
# MAIN
#----------------------

if __name__ == "__main__":
    run()
