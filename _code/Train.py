import numpy as np
import os, time, copy, random
from glob import glob

from torchvision import models, transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import torch.nn as nn
import torch

from .Sampler import BalanceSampler
from .Reader import ImageReader
from .Utils import invDict
from scipy.stats import special_ortho_group

import matplotlib.pyplot as plt     # for data inspection

def norml2(vec):# input N by F
    F = vec.size(1)
    w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F,1))))
    return vec.div(w)

PHASE = ['tra','val']

class learn():
    def __init__(self, no, idx, dst, gpuid, RGBmean, RGBstdv, data_dict, num_epochs=10, init_lr=0.01, decay=0.01, batch_size=256, imgsize=256, avg=4, num_workers=16, accImg="acc.png", lossImg="loss.png", bestEpochFile="epoch.txt", preTrained=True):
        self.no = no
        self.idx = idx
        self.dst = dst
        self.gpuid = gpuid
        
        if len(gpuid)>1: 
            self.mp = True
        else:
            self.mp = False
            
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.decay_time = [False,False]
        self.init_lr = init_lr
        self.decay_rate = decay
        self.num_epochs = num_epochs

        self.avg = avg
        self.data_dict = data_dict
        
        self.imgsize = imgsize
        self.RGBmean = RGBmean
        self.RGBstdv = RGBstdv
        
        self.accImg = accImg
        self.lossImg = lossImg
        self.bestEpochFile = bestEpochFile
        
        self.preTrained = preTrained
        
        self.record = []
        if not self.setsys(): print('system error'); return

    def run(self):
        self.loadData()
        self.setModel()
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.BCELoss()
        #self.criterion = nn.MSELoss()
        self.opt(self.num_epochs)
        self.saveFigLossAcc(self.accImg, self.lossImg)
        return
    

    ##################################################
    # step 0: System check
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        torch.cuda.set_device(self.gpuid[0]); print('Current device is GPU: {}'.format(torch.cuda.current_device()))
        return True
    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        self.data_transforms = {'tra': transforms.Compose([
                                       transforms.Resize(int(self.imgsize*1.1)),
                                       #transforms.RandomCrop(self.imgsize),
                                       #transforms.RandomHorizontalFlip(),
                                       #transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(self.RGBmean, self.RGBstdv)]),
                                'val': transforms.Compose([
                                       transforms.Resize(self.imgsize),
                                       #transforms.CenterCrop(self.imgsize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(self.RGBmean, self.RGBstdv)])}
        

        self.dsets = {p: ImageReader(self.data_dict[p], self.data_transforms[p]) for p in PHASE}
        self.intervals = {p:self.dsets[p].intervals for p in PHASE}
        self.classSize = len(self.intervals['tra'])
        print('output size: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        print('Setting model')
        self.model = models.resnet18(pretrained=self.preTrained)
        
        for param in self.model.parameters(): 
            param.requires_grad = False
            
        num_ftrs = self.model.fc.in_features
        #print(num_ftrs)
        self.model.fc = nn.Linear(num_ftrs, self.classSize)
        self.model.avgpool=nn.AvgPool2d(self.avg)
        self.model.conv2_drop = nn.Dropout2d()
        self.model.Dropout = nn.Dropout(0.1)
        
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.init_lr, momentum=0.9)

        return

    def lr_scheduler(self, epoch):
        if epoch>0.6*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.9*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self):
        if self.mp:
            self.model.module.train(True)  # Set model to training mode
        else:
            self.model.train(True)  # Set model to training mode
            
        dataLoader = torch.utils.data.DataLoader(self.dsets['tra'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        L_data, T_data, N_data = 0.0, 0, 0
        # iterate batch
        for data in dataLoader:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                #print(inputs_bt.shape)
                #print(labels_bt)
                fvec = self.model((inputs_bt.cuda()))
                #print(fvec)
                #argmax, _ = fvec.max(1)
                #print(argmax)
                loss = self.criterion(fvec, (labels_bt).cuda())

            loss.backward()
            self.optimizer.step()  

            _, preds_bt = torch.max(fvec.cpu(), 1)
            
            L_data += loss.item()
            T_data += torch.sum(preds_bt == labels_bt).item()
            N_data += len(labels_bt)
          
        return L_data/N_data, T_data/N_data 

    def val(self):
        if self.mp:
            self.model.module.train(False)  # Set model to training mode
        else:
            self.model.train(False)  # Set model to training mode
            
        dataLoader = torch.utils.data.DataLoader(self.dsets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        L_data, T_data, N_data = 0.0, 0, 0
        # iterate batch
        for data in dataLoader:
            inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
            
            fvec = self.model((inputs_bt.cuda()))
            
            loss = self.criterion(fvec,(labels_bt).cuda())
            
            _, preds_bt = torch.max(fvec.cpu(), 1)

            L_data += loss.item()
            T_data += torch.sum(preds_bt == labels_bt).item()
            N_data += len(labels_bt)
            
        return L_data/N_data, T_data/N_data
        
    def opt(self, num_epochs):
        # recording time and epoch acc and best result
        since = time.time()
        self.best_epoch = 0
        self.best_acc = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
            self.lr_scheduler(epoch)
            
            tra_loss, tra_acc = self.tra()
            val_loss, val_acc = self.val()
            
            self.record.append((epoch, tra_loss, val_loss, tra_acc, val_acc))
            print('tra - Loss:{:.4f} - Acc:{:.4f}\nval - Loss:{:.4f} - Acc:{:.4f}'.format(tra_loss, tra_acc, val_loss, val_acc))    
    
            # deep copy the model
            if epoch >= 1 and val_acc> self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model, self.dst + 'model_{:02}_{}.pth'.format(self.idx,self.no))
        
        torch.save(self.record, self.dst + 'record_{:02}_{}.pth'.format(self.idx, self.no))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print('Best val acc in epoch: {}'.format(self.best_epoch))
        
        epochFile = open(self.bestEpochFile,"w")
        epochFile.write('Acc:{:.4f} - Epoch:{:d}'.format(self.best_acc,self.best_epoch))
        epochFile.close()
        
        return
    def saveFigLossAcc(self, accImg, lossImg):
        plt.clf()
        train_loss = [self.record[i][1] for i in range(len(self.record))]
#         print("train loss: ")
#         print(train_loss)
        val_loss   = [self.record[i][2] for i in range(len(self.record))]
#         print("val loss: ")
#         print(val_loss)
        epochs = np.arange(1, len(self.record) + 1)

        t_line = plt.plot(epochs, train_loss, label="Train")
        v_line = plt.plot(epochs, val_loss, label="Val")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(lossImg)
        
        train_acc = [self.record[i][3] for i in range(len(self.record))]
#         print("train acc: ")
#         print(train_acc)
        val_acc   = [self.record[i][4] for i in range(len(self.record))]
#         print("val acc: ")
#         print(val_acc)

        plt.clf()
        t_line = plt.plot(epochs, train_acc, label="Train")
        v_line = plt.plot(epochs, val_acc, label="Val")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(accImg)
