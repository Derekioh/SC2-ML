import os, torch, random

from torch.utils.data.sampler import SequentialSampler
from torch.autograd import Variable
from torchvision import transforms

from .Utils import norml2
from .Reader import ImageReader
from .color_lib import RGBmean,RGBstdv

def eva(dsets, model):
    Fvecs = []# 256-1200/512-400
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=400, sampler=SequentialSampler(dsets), num_workers=48)

    for data in dataLoader:
        inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
        fvec = norml2(model(Variable(inputs_bt.cuda(),volatile=True)))
        fvec = fvec.data.cpu()
        
        for i in range(fvec.size(0)): 
            Fvecs.append(fvec[i,:].view(-1))
            
    return torch.stack(Fvecs,0)


def RunTest(Data, dst, data_dict, imgsize):
    data_transforms = transforms.Compose([transforms.Resize(imgsize),
                                          transforms.CenterCrop(imgsize),
                                          transforms.ToTensor(),
                                          transforms.Normalize(RGBmean[Data], RGBstdv[Data])])

    dsets = {p: ImageReader(data_dict[p], data_transforms) for p in data_dict}
    torch.save(dsets, dst + 'dsets.pth')
    
    model = torch.load(dst + 'model_{:02}_{}.pth'.format(0,0)).train(False)

    Fvecs = eva(dsets['tra'], model)
    torch.save(Fvecs, dst + 'traFvecs.pth')
    Fvecs = eva(dsets['val'], model)
    torch.save(Fvecs, dst + 'valFvecs.pth')

    
    
    
    