#----------------------
# USER SPECIFIED DATA
#----------------------

import numpy as np
import pandas as pd
import os
import natsort
import cv2
import copy

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

import preprocess as pp
import Net_Fcn_Mod as nf

num_classes = 2
image_shape = (332, 514)
num_epochs = 100
batch_size = 1
use_gpu = True

data_dir = "../data"
runs_dir = "../output"
base_dir = "../data/all_navcam"

UseValidationSet = False
UseBaseSet = True

scale_percent = 120
w = int(image_shape[1]*scale_percent/100)
h = int(image_shape[0]*scale_percent/100)
dim = (w, h)

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
        return tensor_image, idx, self.total_imgs[idx]

def load_data():
    #Define transforms
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    t_transform = transforms.Compose([transforms.ToTensor(), normalize])
    l_transform = transforms.Compose([transforms.ToTensor()])

    t_set = RockDataSet(base_dir + '/left_jpgs', t_transform)
    lt_set = RockDataSet(base_dir + '/labels_pngs', l_transform)

    #Set up loader for both train and validation sets
    t_loader = DataLoader(t_set, batch_size = batch_size, shuffle = True)
    l_loader = DataLoader(lt_set, batch_size = batch_size, shuffle = False)

    datasets = {'train_label': lt_set, 'train': t_set}
    loaders = {'train': t_loader, 'label': l_loader}

    return datasets, loaders

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

    '''for bbox in bc:
        h, w = np.shape(image)
        if bbox['i_max'] == h:
            bbox['i_max'] = h-1
        if bbox['j_max'] == w:
            bbox['j_max'] = w-1

        r[bbox['i_min'], bbox['j_min']:bbox['j_max']+1] = 0
        r[bbox['i_max'], bbox['j_min']:bbox['j_max']+1] = 0
        r[bbox['i_min']:bbox['i_max']+1, bbox['j_min']] = 0
        r[bbox['i_min']:bbox['i_max']+1, bbox['j_max']] = 0

        g[bbox['i_min'], bbox['j_min']:bbox['j_max']+1] = 100
        g[bbox['i_max'], bbox['j_min']:bbox['j_max']+1] = 100
        g[bbox['i_min']:bbox['i_max']+1, bbox['j_min']] = 100
        g[bbox['i_min']:bbox['i_max']+1, bbox['j_max']] = 100

        b[bbox['i_min'], bbox['j_min']:bbox['j_max']+1] = 100
        b[bbox['i_max'], bbox['j_min']:bbox['j_max']+1] = 100
        b[bbox['i_min']:bbox['i_max']+1, bbox['j_min']] = 100
        b[bbox['i_min']:bbox['i_max']+1, bbox['j_max']] = 100
        print("heyo")
    '''

    rgb = np.stack([r, g, b], axis = 2)

    return rgb

def compute_bbox_coordinates(mask_img, lookup_range, verbose, curItr, itrList):

    bbox_list = list()


    img = Image.fromarray(np.uint8(mask_img*255), 'L')
    thresh = cv2.threshold(np.array(img),128,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        #count = 0
        #score = 0
        M = cv2.moments(cntr)
        if (M["m00"] == 0):
            if verbose == 1:
                itrList.append(curItr)
                return 0
        else:
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
        bbox = {'i_min': y, 'j_min': x, 'i_max': y+h, 'j_max': x+w, 'cX': cX, 'cY': cY}
        '''
        arr = (probs[1][y:y+h,x:x+w]).flatten()
        scale = 3
        comp = int(len(arr)/scale) if len(arr) >= scale else 1
        idx = arr.argsort()[-comp:]
        score = float(sum(arr[idx])/comp)
        #for pxl_i in range(y, y+h):
        #    for pxl_j in range(x, x+w):
        #        #if probs[1][pxl_i][pxl_j] > probs[0][pxl_i][pxl_j]:
        #        score += probs[1][pxl_i][pxl_j]
        #        count += 1
        #score = 0 if count == 0 else float(score) / count
        max = 0.6
        score = (score-0.5)/(max-0.5)
        if score > 1.0:
            score = 1.0
        bbox = {'i_min': y, 'j_min': x, 'i_max': y+h, 'j_max': x+w, 'score': score}
        '''
        bbox_list.append(bbox)
    return bbox_list

def blob_overlay(im_orig, bbox_orig, im_big, bbox_big):

    mY, mX = im_orig.shape
    finIm = copy.deepcopy(im_orig)

    for k in range(len(bbox_big)):
        curOr = bbox_orig[k]
        curBig = bbox_big[k]

        '''
        dLX = curBig['cX']- curBig['j_min']
        dUX = curBig['j_max']- curBig['cX']
        dLY = curBig['cY']- curBig['i_min']
        dUY = curBig['i_max']- curBig['cY']
        if dLX > curOr['cX']:
            dLX = curOr['cX']
        if dLY > curOr['cY']:
            dLY = curOr['cY']
        if dUX > (mX-curOr['cX']-1):
            dUX = mX-curOr['cX']-1
        if dUY > (mY-curOr['cY']-1):
            dUY = mY-curOr['cY']-1
        finIm[curOr['cY']- dLY:curOr['cY']+ dUY, curOr['cX']- dLX:curOr['cX']+ dUX] = \
            im_big[curBig['cY']- dLY:curBig['cY']+ dUY, curBig['cX']- dLX:curBig['cX']+ dUX]
        '''

        for j in range(curBig['j_min'],curBig['j_max']):
            for i in range(curBig['i_min'],curBig['i_max']):
                adjI = i-(curBig['cY']- curOr['cY'])
                adjJ = j-(curBig['cX']- curOr['cX'])
                if (0 <= adjI < mY) and (0 <= adjJ < mX):
                    if im_big[i][j] == 1:
                        finIm[adjI][adjJ] = 1

    return finIm

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


    #Training
    setLen = len(datasets['train'])
    avgLoss = -1
    if UseBaseSet:
        #Loss = DiceLoss()
        #baseItr = iter(loaders['label'])
        itrList = list()
        for itr in range(setLen):
            if itr < 3003:
                pass
            else:
                im0, ind, pre = datasets['train_label'][itr]
                im = cv2.imread(base_dir + "/labels_pngs/" + pre)
                #im = Image.fromarray(np.uint8(imBase), 'L')
                #im, ind = next(baseItr)
                #plt.imshow(im[0,:,:,:].permute(1,2,0))
                #plt.savefig('imnorm.png')
                #plt.imshow(datasets['orig'][ind][0].permute(1,2,0))
                #plt.savefig('imorig.png')
                #lab = datasets['train_label'][ind][0]
                #plt.imshow(lab.permute(1,2,0))
                #plt.savefig('imlab.png')
                resize = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
                resize = torch.from_numpy(resize).permute(2,0,1)
                out0 = pp.labelConvert(im0, num_classes)
                out = pp.labelConvert(resize, num_classes)
                om0 = out0[0][1]
                #print(om0.sum())
                om = out[0][1]
                #print(om.shape)
                coordBase0 = om0 == 1
                coordBase = om == 1
                bbox_orig = compute_bbox_coordinates(coordBase0, 5, 0, itr, itrList)
                bbox_big = compute_bbox_coordinates(coordBase, 5, 0, itr, itrList)
                fin = blob_overlay(om0, bbox_orig, om, bbox_big)
                #print(fin.sum())
                coordBaseF = fin == 1
                bbox_fin = compute_bbox_coordinates(coordBaseF, 5, 1, itr, itrList)
                if bbox_fin != 0:
                    rgb = decode_segmap(fin, bbox_fin)
                    plt.imshow(rgb)
                    name = pre[:-4] + '_resize' + '.png'
                    plt.savefig(base_dir + '/labels_resize/' + name)
            print("Image "+str(itr+1)+"/"+str(setLen)+" saved")


    torch.cuda.empty_cache()
    print(itrList)

    return


#----------------------
# MAIN
#----------------------

if __name__ == "__main__":
    run()
