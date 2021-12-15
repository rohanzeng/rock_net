#Preprocess labeled images of red and green pixels into one hot labels

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.misc as misc
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

BlueMean = 103.939
GreenMean = 116.779
RedMean = 123.68

def labelConvert(label, NumClasses):
    #label = label.cuda()
    rgb, h, w = label.size()
    batchsize = 1
    target = torch.zeros(batchsize, NumClasses, h, w) #.cuda()

    #White layer is taken care of since it's R value > 0
    target[0][0][label[0] > 0] = 1 #Converts red layer to non rock, converts white layer to non rock
    target[0][1][label[1] > 0] = 1 #Converts green layer to rock
    '''print(sum(sum(target[0][0].numpy())))
    print(sum(sum(target[0][1].numpy())))
    print(sum(sum(target[0][0].numpy()))+sum(sum(target[0][1].numpy())))
    print(h*w)'''

    return torch.autograd.Variable(target, requires_grad = False)
