import os, torch, random

from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable
from torchvision import transforms

from .Bc_utils import ImageReader
from .color_lib import RGBmean,RGBstdv

def eva(dsets, model):
    Fvecs = []# 800/500/400
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=400, sampler=SequentialSampler(dsets), num_workers=24)

    T_data, N_data = 0,0
    for data in dataLoader:
        inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
        fvec = model(Variable(inputs_bt.cuda(),volatile=True))
        fvec = fvec.data.cpu()
        
        _, preds_bt = torch.max(fvec, 1)
        
        T_data += torch.sum(preds_bt.view(-1) == labels_bt)
        N_data += len(labels_bt)
        
        for i in range(fvec.size(0)): 
            Fvecs.append(fvec[i,:].view(-1))
            
    Fvecs = torch.stack(Fvecs,0)
    return Fvecs, T_data/N_data 


def RunTest(Data, dst, data_dict, imgsize=256):
    
    data_transforms = transforms.Compose([transforms.Resize(imgsize),
                                          transforms.CenterCrop(imgsize),
                                          transforms.ToTensor(),
                                          transforms.Normalize(RGBmean[Data], RGBstdv[Data])])

    dsets = {p: ImageReader(data_dict[p], data_transforms) for p in data_dict}
    torch.save(dsets, dst + 'dsets.pth')
    
    model = torch.load(dst + 'model_{:02}_{}.pth'.format(0,0)).train(False)

    Fvecs, acc_d = eva(dsets['tra'], model)
    torch.save(Fvecs, dst + 'traFvecs.pth')
    Fvecs, acc_d = eva(dsets['val'], model)
    torch.save(Fvecs, dst + 'valFvecs.pth')

    