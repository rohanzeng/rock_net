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
    '''i1 = -100
    i2 = -100
    i3 = -100
    i4 = -100
    i5 = -100
    i6 = -100
    for k in range(rgb):
        for j in range(w):
            for i in range(h):
                if k == 0:
                    if i1 == -100:
                        i1 = label[0][i][j]
                        i2 = label[0][i][j]
                        col1 = (label[0][i][j], label[1][i][j], label[2][i][j])
                    if i2 == i1:
                        i2 = label[0][i][j]
                        col2 = (label[0][i][j], label[1][i][j], label[2][i][j])
                if k == 1:
                    if i3 == -100:
                        i3 = label[0][i][j]
                        i4 = label[0][i][j]
                        col3 = (label[0][i][j], label[1][i][j], label[2][i][j])

                    if i4 == i3:
                        i4 = label[0][i][j]
                        col4 = (label[0][i][j], label[1][i][j], label[2][i][j])

                if k == 2:
                    if i5 == -100:
                        i5 = label[0][i][j]
                        i6 = label[0][i][j]
                        col5 = (label[0][i][j], label[1][i][j], label[2][i][j])

                    if i6 == i5:
                        i6 = label[0][i][j]
                        col6 = (label[0][i][j], label[1][i][j], label[2][i][j])

    print('i1 = ' + str(i1))
    print('i2 = ' + str(i2))
    print('i3 = ' + str(i3))
    print('i4 = ' + str(i4))
    print('i5 = ' + str(i5))
    print('i6 = ' + str(i6))

    print('col1 = ' + str(col1))
    print('col2 = ' + str(col2))
    print('col3 = ' + str(col3))
    print('col4 = ' + str(col4))
    print('col5 = ' + str(col5))
    print('col6 = ' + str(col6))
    '''
    '''for c in range(NumClasses):
        for b in range(batchsize):
            #print(label[b])
            #print(c)
            if c == 0:
                target[b][c][label[0] > 0] = 1
            else:
                target[b][c][label[1] > 0] = 1'''

    #White layer is taken care of since it's R value > 0
    target[0][0][label[0] > 0] = 1 #Converts red layer to non rock, converts white layer to non rock
    target[0][1][label[1] > 0] = 1 #Converts green layer to rock
    '''print(sum(sum(target[0][0].numpy())))
    print(sum(sum(target[0][1].numpy())))
    print(sum(sum(target[0][0].numpy()))+sum(sum(target[0][1].numpy())))
    print(h*w)'''

    return torch.autograd.Variable(target, requires_grad = False)
