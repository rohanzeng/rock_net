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
from sklearn.metrics import jaccard_similarity_score as jsc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
import glob
import shutil
#from skimage import io
#import helper

import preprocess as pp
import Net_Fcn_Mod as nf

num_classes = 2
image_shape = (332, 514)
num_epochs = 100
batch_size = 1
a = 1e-4
weight_decay = 1e-4
use_gpu = True
cuda = torch.device('cuda:1')

data_dir = "../data"
runs_dir = "../output"
aug_dir = "../data/all_navcam"
base_dir = "../data/all_navcam/output"
label_dir = "../data/all_navcam/outputL"
vgg_path = "../data/rvgg"

clusters = [0,17]
clus_dir = "../data/all_navcam/outputC"
numClus = 20

UseValidationSet = False
UseBaseSet = True
UseAugmentedSet = False
TrainedModelWeightDir = "TrainedModelWeights/"
Trained_model_path = '' #TrainedModelWeightDir+"17006.torch"
TrainLossTxtFile = TrainedModelWeightDir + "TrainLossGPU.txt"
ValidLossTxtFile = TrainedModelWeightDir + "ValidLossGPU.txt"
Pretrained_Encoder_Weights = "densenet_cosine_264_k32.pth"

def clusterDir():
    for file in os.listdir(clus_dir+"/train"):
        os.remove(os.path.join(clus_dir+"/train", file))
    for file in os.listdir(clus_dir+"/label"):
        os.remove(os.path.join(clus_dir+"/label", file))
    for i in clusters:
        for file in glob.glob(clus_dir+"/clusters"+str(numClus)+"/cluster"+str(i)+"/*.jpg"):
            curIm = cv2.imread(file)
            #print(curIm)
            #print(file[-28:])
            cv2.imwrite(clus_dir+"/train/"+file[-28:], curIm)
        for file in glob.glob(clus_dir+"/clusters"+str(numClus)+"/clusterL"+str(i)+"/*.png"):
            curIm = cv2.imread(file)
            cv2.imwrite(clus_dir+"/label/"+file[-28:], curIm)
    return

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

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #inputs = torch.sigmoid(inputs)

        #m = nn.Softmax(dim = 0)

        inputs = inputs[0].view(-1)
        targets = targets[0].view(-1)

        intersection = (inputs*targets).sum()
        total = (inputs+targets).sum()
        union = total-intersection

        #intersection = torch.logical_and(targets, inputs)
        #union = torch.logical_or(targets, inputs)
        IoU = (intersection+smooth)/(union+smooth)

        return 1-IoU

class DiceLoss(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth = 1):

        inputs = inputs[0].view(-1)
        targets = targets[0].view(-1)

        intersection = (inputs*targets).sum()
        dice = (2.*intersection+smooth)/((inputs*inputs).sum() + (targets*targets).sum() + smooth)

        return 1-dice

def load_data():
    #Define transforms
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    t_transform = transforms.Compose([transforms.ToTensor(), normalize])
    l_transform = transforms.Compose([transforms.ToTensor()])

    t_set = RockDataSet(clus_dir + '/train', t_transform)
    ta_set = RockDataSet(aug_dir + '/aug_train', t_transform)
    v_set = RockDataSet(base_dir + '/val/left_jpgs', t_transform)
    te_set = RockDataSet(base_dir + '/test/left_jpgs', t_transform)
    lt_set = RockDataSet(clus_dir + '/label', l_transform)
    lta_set = RockDataSet(aug_dir + '/aug_label', l_transform)
    lv_set = RockDataSet(label_dir + '/val/labels_pngs', l_transform)
    lte_set = RockDataSet(label_dir + '/test/labels_pngs', l_transform)

    #Set up loader for both train and validation sets
    t_loader = DataLoader(t_set, batch_size = batch_size, shuffle = True)
    ta_loader = DataLoader(ta_set, batch_size = batch_size, shuffle = True)
    v_loader = DataLoader(v_set, batch_size = batch_size, shuffle = False)
    te_loader = DataLoader(te_set, batch_size = batch_size, shuffle = True)

    datasets = {'train_label': lt_set, 'valid_label': lv_set, 'aug_label': lta_set, 'train': t_set, 'aug': ta_set, 'valid': v_set}
    loaders = {'train': t_loader, 'valid': v_loader, 'test': te_loader, 'aug': ta_loader}

    return datasets, loaders

def load_model():
    '''
    model = models.vgg16(pretrained = True)
    #print(model)

    #Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Sequential(
                          nn.Linear(4096, 256),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(256, num_classes),
                          nn.LogSoftmax(dim=1))
    '''
    #print(model)
    #if use_gpu:
    #    model = model.to('cuda')

    model = nf.Net(NumClasses = num_classes, PreTrainedModelPath = Pretrained_Encoder_Weights, UseGPU = True,
            UpdateEncoderBatchNormStatistics = True)

    #if use_gpu:
        #model = model.to('cuda')
    #model = model.cuda()

    if not Trained_model_path == "":
        model.load_state_dict(torch.load(Trained_model_path))

    return model

def run():

    clusterDir()
    print("Organized Cluster Data")

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
    model = model.to('cuda')
    optimizer = torch.optim.Adam(params=model.parameters(), lr = a, weight_decay = weight_decay)

    print('Model Data Loaded')

    #Training
    num = 11
    avgLoss = -1
    if UseBaseSet:
        #Loss = DiceLoss()
        baseItr = iter(loaders['train'])
        for itr in range(len(datasets['train'])):
            im, ind = next(baseItr)
            #plt.imshow(im[0,:,:,:].permute(1,2,0))
            #plt.savefig('imnorm.png')
            #plt.imshow(datasets['orig'][ind][0].permute(1,2,0))
            #plt.savefig('imorig.png')
            lab = datasets['train_label'][ind][0]
            #plt.imshow(lab.permute(1,2,0))
            #plt.savefig('imlab.png')
            OneHotLabels = pp.labelConvert(lab, num_classes).to('cuda')
            prob, lb = model.forward(im.permute(0,2,3,1))#.to('cuda'))
            #print(prob.size())
            #print(OneHotLabels.size())
            model.zero_grad()
            loss = -torch.mean((OneHotLabels * torch.log(prob + 0.0000001)))

            #loss = Loss(prob, OneHotLabels)

            if avgLoss == -1:
                avgLoss = float(loss.data.cpu().numpy())
            else:
                avgLoss = avgLoss*0.99+0.01*float(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()

            if ((itr-num) % 100 == 0 and itr > num) or (itr == len(datasets['train'])-1):
                print("Saving Model to file in " + TrainedModelWeightDir)
                torch.save(model.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
                print("model saved")

            if itr % 100 == 0:
                torch.cuda.empty_cache()
                print("Step "+str(itr)+"\n" +
                    "Train Loss="+str(float(loss.data.cpu().numpy()))+"\n" +
                    "Running Average Loss="+str(avgLoss))
                with open(TrainLossTxtFile, "a") as f:
                    f.write("\n"+str(itr)+"\t"+str(float(loss.data.cpu().numpy()))+"\t"+str(avgLoss))
                    f.close()

            #Validation
            if UseValidationSet and itr % 3000 == 0 and itr != 0:
                SumLoss=np.float64(0.0)
                NBatches = int(len(datasets['valid']))
                print("Calculating Validation on " + str(NBatches) + " Images")
                valItr = iter(loaders['valid'])
                for i in range(NBatches):# Go over all validation images
                    im, ind = next(valItr)
                    lab = datasets['valid_label'][ind][0]
                    OneHotLabels = pp.labelConvert(lab, num_classes).to('cuda')
                    prob, lb = model.forward(im.permute(0,2,3,1))
                    TLoss = -torch.mean((OneHotLabels * torch.log(prob + 0.0000001)))
                    SumLoss+=float(TLoss.data.cpu().numpy())
                    NBatches+=1
                SumLoss /= NBatches
                print("Validation Loss: "+str(SumLoss))
                with open(ValidLossTxtFile, "a") as f: #Write validation loss to file
                    f.write("\n" + str(itr) + "\t" + str(SumLoss))
                    f.close()

    if UseAugmentedSet:
        if not UseBaseSet:
            itr = 0
        augItr = iter(loaders['aug'])
        for itra in range(len(datasets['aug'])):
            im, ind = next(augItr)
            lab = datasets['aug_label'][ind][0]
            OneHotLabels = pp.labelConvert(lab, num_classes).to('cuda')
            prob, lb = model.forward(im.permute(0,2,3,1))#.to('cuda'))
            model.zero_grad()
            loss = -torch.mean((OneHotLabels * torch.log(prob + 0.0000001)))
            if avgLoss == -1:
                avgLoss = float(loss.data.cpu().numpy())
            else:
                avgLoss = avgLoss*0.99+0.01*float(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()

            adjItr = itr+itra+1

            if (adjItr-num) % 1000 == 0 and adjItr > num:
                print("Saving Model to file in " + TrainedModelWeightDir)
                torch.save(model.state_dict(), TrainedModelWeightDir + "/" + str(adjItr) + ".torch")
                print("model saved")

            if adjItr % 100 == 0:
                torch.cuda.empty_cache()
                print("Step "+str(adjItr)+"\n" +
                    "Train Loss="+str(float(loss.data.cpu().numpy()))+"\n" +
                    "Running Average Loss="+str(avgLoss))
                with open(TrainLossTxtFile, "a") as f:
                    f.write("\n"+str(adjItr)+"\t"+str(float(loss.data.cpu().numpy()))+"\t"+str(avgLoss))
                    f.close()

            #Validation
            if UseValidationSet and adjItr % 3000 == 0 and itr != 0:
                SumLoss=np.float64(0.0)
                NBatches = int(len(datasets['valid']))
                print("Calculating Validation on " + str(NBatches) + " Images")
                valItr = iter(loaders['valid'])
                for i in range(NBatches):# Go over all validation images
                    im, ind = next(valItr)
                    lab = datasets['valid_label'][ind][0]
                    OneHotLabels = pp.labelConvert(lab, num_classes).to('cuda')
                    prob, lb = model.forward(im.permute(0,2,3,1))
                    TLoss = -torch.mean((OneHotLabels * torch.log(prob + 0.0000001)))
                    SumLoss+=float(TLoss.data.cpu().numpy())
                    NBatches+=1
                SumLoss /= NBatches
                print("Validation Loss: "+str(SumLoss))
                with open(ValidLossTxtFile, "a") as f: #Write validation loss to file
                    f.write("\n" + str(adjItr) + "\t" + str(SumLoss))
                    f.close()
        print("Finished Aug Set")

    torch.cuda.empty_cache()

    return


#----------------------
# MAIN
#----------------------

if __name__ == "__main__":
    run()
