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
import torch.nn.functional as F
#from skimage import io
#import helper

num_classes = 2
image_shape = (332, 514)
num_epochs = 100
batch_size = 1
a = 1e-4
weight_decay = 1e-4
use_gpu = True
set_len = 10
thresh = 0.5

data_dir = "../../data"
runs_dir = "./outputs"
training_dir = "../../data/all_navcam"
vgg_path = "../../data/rvgg"

UseValidationSet = False
TrainedModelWeightDir = "TrainedPyWeights/"
Trained_model_path = TrainedModelWeightDir+"res1012.torch"
TrainLossTxtFile = TrainedModelWeightDir + "TrainLoss.txt"
ValidLossTxtFile = TrainedModelWeightDir + "ValidLoss.txt"

modNam = "res101"

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
    #resize = transforms.Resize(256)
    #crop = transforms.CenterCrop(224)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    t_transform = transforms.Compose([transforms.ToTensor(), normalize])
    l_transform = transforms.Compose([transforms.ToTensor()])

    #Create labeled [l] and train [t] dataset classes, split into train and validation sets
    l_set = RockDataSet(training_dir + '/labels_pngs', l_transform)
    vis_set = RockDataSet(training_dir + '/left_jpgs', t_transform)
    orig_set = RockDataSet(training_dir + '/left_jpgs', l_transform)

    #Set up loader for both train and validation sets
    vis_loader = DataLoader(vis_set, batch_size = batch_size, shuffle = False)

    datasets = {'visual': vis_set, 'label': l_set, 'orig': orig_set}
    loaders = {'visual': vis_loader}

    return datasets, loaders

def load_model():

    #model = nf.Net(NumClasses = num_classes, PreTrainedModelPath = Pretrained_Encoder_Weights, UseGPU = use_gpu,
    #UpdateEncoderBatchNormStatistics = True)

    #model = models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 2, aux_loss = None)

    if modNam == "res50":
        model = models.segmentation.fcn_resnet50(pretrained=False,progress=True,num_classes=2,aux_loss=None)
    elif modNam == "res101":
        model = models.segmentation.fcn_resnet101(pretrained=False,progress=True,num_classes=2,aux_loss=None)
    elif modNam == "deep50":
        model = models.segmentation.deeplabv3_resnet50(pretrained=False,progress=True,num_classes=2,aux_loss=None)
    elif modNam == "deep101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=False,progress=True,num_classes=2,aux_loss=None)
    elif modNam == "deepv3":
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False,progress=True,num_classes=2,aux_loss=None)
    else:
        model = models.segmentation.lraspp_mobilenet_v3_large(pretrained=False,progress=True,num_classes=2,aux_loss=None)

    if use_gpu:
        model.load_state_dict(torch.load(Trained_model_path))
    else:
        model.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))

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

    for bbox in bc:
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

    rgb = np.stack([r, g, b], axis = 2)

    return rgb

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
        bbox_list.append(bbox)
    return bbox_list

'''def compute_bbox_coordinates(mask_img, lookup_range, verbose = 0):

    bbox_list = list()
    visited_pixels = list()

    bbox_found = 0

    for i in range(mask_img.shape[0]):
        for j in range(mask_img.shape[1]):

            if mask_img[i,j] == 1 and (i, j) not in visited_pixels:

                bbox_found += 1

                pixels_to_visit = list()

                bbox = {'i_min': i, 'j_min': j, 'i_max' : i, 'j_max':j}

                pxl_i = i
                pxl_j = j

                while True:
                    visited_pixels.append((pxl_i, pxl_j))
                    bbox['i_min'] = min(bbox['i_min'], pxl_i)
                    bbox['j_min'] = min(bbox['j_min'], pxl_j)
                    bbox['i_max'] = max(bbox['i_max'], pxl_i)
                    bbox['j_max'] = max(bbox['j_max'], pxl_j)

                    i_min = max(0, pxl_i - lookup_range)
                    j_min = max(0, pxl_j - lookup_range)

                    i_max = min(mask_img.shape[0], pxl_i + lookup_range + 1)
                    j_max = min(mask_img.shape[1], pxl_j + lookup_range + 1)

                    for i_k in range(i_min, i_max):
                         for j_k in range(j_min, j_max):

                            if mask_img[i_k, j_k] == 1 and (i_k, j_k) not in visited_pixels and (\
                            i_k, j_k) not in pixels_to_visit:
                                pixels_to_visit.append((i_k, j_k))
                                visited_pixels.append((i_k, j_k))

                    if not pixels_to_visit:
                        break

                    else:
                        pixel = pixels_to_visit.pop()
                        pxl_i = pixel[0]
                        pxl_j = pixel[1]

                bbox_list.append(bbox)
    if verbose:
        print("BBOX Found: %d" % bbox_found)

    return bbox_list
'''

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
    model.cuda()

    print('Model Data Loaded')

    #Visualizing
    it = iter(loaders['visual'])
    for itr in range(set_len):
        im, ind, pre = next(it)
        #im = im.permute(0,2,3,1)
        out = F.softmax(model(im.cuda())['out'], dim = 1)
        a = out.detach().cpu().numpy()[0,:,:,:]
        #print(a[0,:,:])
        #print(a[1,:,:])
        #om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        #print(om)
        #print(om.shape)
        #coordBase = om == 1
        omBase = out.squeeze().detach().cpu().numpy()
        #coordBase = np.zeros((omBase.shape[1], omBase.shape[2]))
        coordBase = ((omBase[1,:,:]+0.5-thresh)>(omBase[0,:,:])).astype(int)
        bbox_coords = compute_bbox_coordinates(coordBase, a, lookup_range = 5, verbose = 0)
        rgb = decode_segmap(coordBase, bbox_coords)
        base = pre[0]
        plt.imshow(rgb)
        net = 'im' + str(itr) + '_net.png' #base[:-4] + '_net' + '.png'
        plt.savefig(runs_dir + '/' + net)
        plt.imshow(datasets['orig'][ind][0].permute(1,2,0))
        orig = 'im' + str(itr) + '_orig.png' #base[:-4] + '_orig' + '.png'
        plt.savefig(runs_dir + '/' + orig)
        plt.imshow(datasets['label'][ind][0].permute(1,2,0))
        lab = 'im' + str(itr) + '_lab.png' #base[:-4] + '_lab' + '.png'
        plt.savefig(runs_dir + '/' + lab)
        print('image ' + str(itr) + '/' + str(set_len) + ' saved')

        #coordBase = om == 1
        #bbox_coords = compute_bbox_coordinates(coordBase, lookup_range = 1, verbose = 0)
        #print(bbox_coords)
    return


#----------------------
# MAIN
#----------------------

if __name__ == "__main__":
    run()
