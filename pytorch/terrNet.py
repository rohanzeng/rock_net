# Use a pretrained Pytorch Network to perform KMeans clustering to cluster images in a folder

#Packages
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import cv2
import os
import glob
import shutil

#Parameters
input_dir = './../data/all_navcam/left_jpgs'
glob_dir = input_dir+'/*.jpg'
label_dir = './../data/all_navcam/labels_pngs'
globL_dir = label_dir+'/*.png'
k = 18
saveImages = True
numIm = 50

torch.cuda.set_device(0)
torch.cuda.empty_cache()

#Images Setup
images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
totIm = len(images)
intDiv = totIm//numIm
modDiv = totIm%numIm
paths = [file for file in glob.glob(glob_dir)]
pathsL = [file for file in glob.glob(globL_dir)]
images = torch.tensor(np.float32(images).reshape(len(images), -1)/255, requires_grad = False)
images = images.cuda()
print("Images Loaded")

#Model
model = models.mobilenet_v2(pretrained = True) #models.vgg16_bn(pretrained = True)
model.cuda()
model.eval()
print("Model Loaded")

#predictions = model(images.reshape(-1, 224, 224, 3).permute(0,3,1,2))
#predictions = [model(image.reshape(-1, 224, 224, 3).permute(0,3,1,2)) for image in images]
#pred_images = prediction.reshape(numIm, -1)

'''
curInt = 0
totClass = np.zeros(k)
for i in range(intDiv):
    predictions = np.zeros(numIm)
    for j in range(curInt*numIm, curInt*numIm+numIm):
        #print(images[j].shape)
        predictions[j%numIm] = model(images[j].reshape(224, 224, 3).permute(2,0,1).unsqueeze(0))
    pred_images = predictions.reshape(numIm, -1)

    kmodel = KMeans(n_clusters = k, n_jobs = -1, random_state = 728)
    kmodel.fit(pred_images)
    kpredictions = kmodel.predict(pred_images)
    shutil.rmtree('output')

    for i in range(k):
        totClass[i] += (kpredictions == i).sum()
    curInt += 1
    print(str(curInt)+"/"+str(intDiv))

for i in range(modDiv):
    predictions = np.zeros(modDiv)
    for j in range(curInt*numIm, curInt*numIm+modDiv):
        predictions[j%numIm] = model(images[j].reshape(224, 224, 3).permute(2,0,1).unsqueeze(0))
    pred_images = predictions.reshape(numIm, -1)

    kmodel = KMeans(n_clusters = k, n_jobs = -1, random_state = 728)
    kmodel.fit(pred_images)
    kpredictions = kmodel.predict(pred_images)
    shutil.rmtree('output')

    for i in range(k):
        totClass[i] += (kpredictions == i).sum()


for i in range(k):
    #classTots.append((kpredictions == i).sum())
    print("Class "+str(i)+": "+str(totClass[i])+" Images")
'''

# Preprocess the images
with torch.no_grad():
    predictions = torch.zeros([totIm,1,1000])
    count = 0
    for image in images:
        a = model(image.reshape(-1, 224, 224, 3).permute(0,3,1,2))
        #print(a.shape)
        predictions[count] = (model(image.reshape(-1, 224, 224, 3).permute(0,3,1,2)))
        count += 1
    pred_images = predictions.reshape(images.shape[0], -1)
    #torch.save(model.state_dict(), "classNet.torch")
print("Processed Images")


#K Means
kmodel = KMeans(n_clusters = k, n_jobs = -1, random_state = 728)
kmodel.fit(pred_images)
kpredictions = kmodel.predict(pred_images)
shutil.rmtree('./../data/all_navcam/outputC/clusters'+str(k)+'a')

if saveImages:
    for i in range(k):
        os.makedirs("./../data/all_navcam/outputC/clusters"+str(k)+"a"+"/cluster"+str(i))
        os.makedirs("./../data/all_navcam/outputC/clusters"+str(k)+"a"+"/clusterL"+str(i))

    for i in range(len(paths)):
        shutil.copy2(paths[i], "./../data/all_navcam/outputC/clusters"+str(k)+"a"+"/cluster"+str(kpredictions[i]))
        shutil.copy2(pathsL[i], "./../data/all_navcam/outputC/clusters"+str(k)+"a"+"/clusterL"+str(kpredictions[i]))
    print("Saved Images")


#for i in range(len(paths)):
#classTots = []
for i in range(k):
    #classTots.append((kpredictions == i).sum())
    print("Class "+str(i)+": "+str((kpredictions == i).sum())+" Images")




