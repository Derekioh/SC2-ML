from _code._train import RunTrain
from _code._test import RunTest
from glob import glob
import os, random
###### resnet18 256 by 256 ######
Data='starcraft_nofog'
src = 'SC2_data_nofog/'
#src = 'SC2_last30/'
dst = '_result/'

img_size = 256

# all data
data_dict = {os.path.basename(d):glob(d+'/*.png') for d in glob(src+'/*')}

# sep data
data_dict_sep = {p:{k:[] for k in data_dict} for p in ['tra','val']}
for k in data_dict:
    imgs = data_dict[k]
    random.shuffle(imgs)
    LEN = int(len(imgs)*0.8)
#     DELETE_ME = imgs[0:101]
#     data_dict_sep['tra'][k]=DELETE_ME[:80]
#     data_dict_sep['val'][k]=DELETE_ME[80:]
    data_dict_sep['tra'][k]=imgs[:LEN]
    data_dict_sep['val'][k]=imgs[LEN:]

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc1.png", lossImg="loss1.png", bestEpochFile="epoch1.txt", preTrained=True, num_epochs=50, init_lr=0.01)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc2.png", lossImg="loss2.png", bestEpochFile="epoch2.txt", preTrained=False, num_epochs=50, init_lr=0.01)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc3.png", lossImg="loss3.png", bestEpochFile="epoch3.txt", preTrained=True, num_epochs=50, init_lr=0.001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc4.png", lossImg="loss4.png", bestEpochFile="epoch4.txt", preTrained=False, num_epochs=50, init_lr=0.001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc5.png", lossImg="loss5.png", bestEpochFile="epoch5.txt", preTrained=True, num_epochs=50, init_lr=0.001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc6.png", lossImg="loss6.png", bestEpochFile="epoch6.txt", preTrained=True, num_epochs=20, init_lr=0.0001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc8.png", lossImg="loss8.png", bestEpochFile="epoch8.txt", preTrained=True, num_epochs=20, init_lr=0.0001)

RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="acc_nofog.png", lossImg="loss_nofog.png", bestEpochFile="epoch_nofog.txt", preTrained=True, num_epochs=50, init_lr=0.001)

Data='starcraftLast30_nofog'
src = 'SC2_last30_nofog/'

# all data
data_dict = {os.path.basename(d):glob(d+'/*.png') for d in glob(src+'/*')}

# sep data
data_dict_sep = {p:{k:[] for k in data_dict} for p in ['tra','val']}
for k in data_dict:
    imgs = data_dict[k]
    random.shuffle(imgs)
    LEN = int(len(imgs)*0.8)
#     DELETE_ME = imgs[0:101]
#     data_dict_sep['tra'][k]=DELETE_ME[:80]
#     data_dict_sep['val'][k]=DELETE_ME[80:]
    data_dict_sep['tra'][k]=imgs[:LEN]
    data_dict_sep['val'][k]=imgs[LEN:]
    
# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast1.png", lossImg="lossLast1.png", bestEpochFile="epochLast1.txt", preTrained=True, num_epochs=50, init_lr=0.01)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast2.png", lossImg="lossLast2.png", bestEpochFile="epochLast2.txt", preTrained=False, num_epochs=50, init_lr=0.01)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast3.png", lossImg="lossLast3.png", bestEpochFile="epochLast3.txt", preTrained=True, num_epochs=50, init_lr=0.001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast4.png", lossImg="lossLast4.png", bestEpochFile="epochLast4.txt", preTrained=False, num_epochs=50, init_lr=0.001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast5.png", lossImg="lossLast5.png", bestEpochFile="epochLast5.txt", preTrained=True, num_epochs=50, init_lr=0.001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast6.png", lossImg="lossLast6.png", bestEpochFile="epochLast6.txt", preTrained=True, num_epochs=50, init_lr=0.0001)

# RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast8.png", lossImg="lossLast8.png", bestEpochFile="epochLast8.txt", preTrained=True, num_epochs=20, init_lr=0.00001)

RunTrain(Data, dst, data_dict_sep, img_size, bt=60, core=[0], accImg="accLast_nofog.png", lossImg="lossLast_nofog.png", bestEpochFile="epochLast_nofog.txt", preTrained=True, num_epochs=50, init_lr=0.001)