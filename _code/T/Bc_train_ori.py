from .color_lib import RGBmean,RGBstdv
from .B_utils import createID, createID2

from .Bc_learn import learn
from .Bc_test import RunTest

import os, torch

def RunTrain(Data, dst, data_dict, imagesize, bt=128, core=[0,1]):
        
    if not os.path.exists(dst): os.makedirs(dst)

    N = len(data_dict['tra'])
    
    print(N)
    
    x = learn(0, 0, dst, core, RGBmean[Data], RGBstdv[Data], data_dict, init_lr=0.1, decay=[6,3,0.1], batch_size=bt, imgsize=imagesize, avg=8)
    a = x.run()