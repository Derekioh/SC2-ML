from .color_lib import RGBmean,RGBstdv
from .Utils import createID, createID2
from .Train import learn

import os, torch, random

def RunTrain(Data, dst, data_dict, imagesize, bt=16, core=[0], accImg="acc.png", lossImg="loss.png", bestEpochFile="epoch.txt", preTrained=True, num_epochs=50, init_lr=0.001):
    if not os.path.exists(dst): os.makedirs(dst)
    for i in range(1): #5
        learn(0, i, dst, core, RGBmean[Data], RGBstdv[Data], data_dict, num_epochs, init_lr, decay=0.1, batch_size=bt, imgsize=imagesize, avg=8, num_workers=16, accImg=accImg, lossImg=lossImg, bestEpochFile=bestEpochFile, preTrained=preTrained).run() #num_epochs=50




