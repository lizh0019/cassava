package_path = '../input/pytorch-image-models/pytorch-image-models-master' #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
import sys; sys.path.append(package_path)

from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from  torch.cuda.amp import autocast, GradScaler

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2

import timm #from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import log_loss
from scipy.ndimage import gaussian_filter

model_choice = -2 #-2, -1, 0, 1, 2, 3
CFG = {
    'fold_num': 5,
    'n_class': 5,
    'seed': 719,
    'model_arch': ['tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b7_ap', 'resnet50', 'tf_efficientnet_b3_ns'][model_choice],
    'epochs': 10,
    'valid_bs': 16, #6, #32
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 1, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'tta': 3,
    #'used_epochs': [[3,5,6,7,9,13,14,16],[4,5,6,7,9],[],[9],[]], #range(10), #[6,7,8,9],
    'used_epochs': [[6,7,9,14],[4,7,9],[6,7,8,9],[8,9],[]],
    'weights': [1]*20
}

fpn_dim = {'tf_efficientnet_b4_ns':[24,32,56,112,160,272,448], 'tf_efficientnet_b5_ap':[24,40,64,128,176,304,512], 'tf_efficientnet_b6_ap':[32,40,72,144,200,344,576], 'tf_efficientnet_b7_ap':[32,48,80,160,224,384,640], 'resnet50':[256,256,256,512,512,512,512,1024,1024,1024,1024,1024,1024,2048,2048,2048], 'tf_efficientnet_b3_ns':[24,32,48,96,136,232,384]}
resolution = {'tf_efficientnet_b4_ns':512, 'tf_efficientnet_b5_ap':380, 'tf_efficientnet_b6_ap':528, 'tf_efficientnet_b7_ap':600, 'resnet50': 512, 'tf_efficientnet_b3_ns':300}
CFG['img_size_width'] = CFG['img_size_height'] = resolution[CFG['model_arch']]

train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
train.head()


train.label.value_counts()

submission = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
submission.head()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    return im_rgb


class CassavaDataset(Dataset):
    def __init__(
        self, df, data_root, transforms=None, output_label=True
    ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.df.iloc[index]['label']
          
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        
        img  = get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
            
        # do label smoothing
        if self.output_label == True:
            return img, target
        else:
            return img
            
       

from albumentations.pytorch import ToTensorV2

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop, RandomCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


def get_inference_transforms():
    if CFG['img_size_width'] == CFG['img_size_height']:
        #mul_res = [(380,380),(456,456),(528,528)]
        #i_res = 0 #np.random.choice(len(mul_res))
        return Compose([
            #CenterCrop(CFG['img_size_height'], CFG['img_size_width'], p=1.),
            RandomResizedCrop(CFG['img_size_height'], CFG['img_size_width']),
            #RandomResizedCrop(mul_res[i_res][0],mul_res[i_res][1]),
            Transpose(p=0.5),
            #IAAPerspective(p=0.5),
            #CoarseDropout(p=0.5),
            #GridDistortion(p=0.5),
            #OpticalDistortion(p=0.5),
            #IAASharpen(p=0.5),
            #IAAPiecewiseAffine(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
    else:        
        return Compose([
            #CenterCrop(CFG['img_size_height'], CFG['img_size_width'], p=1.),
            #RandomResizedCrop(CFG['img_size_height'], CFG['img_size_width']),
            RandomResizedCrop(mul_res[i_res][0],mul_res[i_res][1]),
            Transpose(p=0.5),
            #IAAPerspective(p=0.5),
            #CoarseDropout(p=0.5),
            #GridDistortion(p=0.5),
            #OpticalDistortion(p=0.5),
            #IAASharpen(p=0.5),
            #IAAPiecewiseAffine(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
        
        
## Model
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.n_class = n_class
        self.model = timm.create_model(model_arch, pretrained=pretrained) #pretrained model
        try:
            self.model.classifier = nn.Linear(self.model.fc.in_features, n_class)
        except:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, n_class) #original decision
        
        self.fpn_sizes = fpn_dim[CFG['model_arch']] #each layer feature depth
        self.model.fpn_classifiers = [nn.Linear(self.fpn_sizes[i], n_class).to(device) for i in range(len(self.fpn_sizes))] #seperate decisions
        
        if model_choice != -2:
            self.ensemble = nn.Linear((1+len(self.fpn_sizes))*n_class, n_class) #ensemble all layers
        else:
            self.ensemble = nn.Linear((0+len(self.fpn_sizes))*n_class, n_class) #ensemble all layers
        self.outputs0 = [] #hooked intermediate features
        try:
            #for name, layer in self.model._modules.items():
            #    layer.register_forward_hook(self.myhook) #record intermediate features
            for i in range(1,5):
                for name, layer in self.model.__getattr__('layer'+str(i))._modules.items():
                    layer.register_forward_hook(self.myhook)
        except:
            for name, layer in self.model.blocks._modules.items():
                layer.register_forward_hook(self.myhook) #record intermediate features
        
    def myhook(self, module, input, output):
        self.outputs0.extend(output) #0-th layer:[0-th image, 1-th, ...], 1-th layer:[0-th image, 1-th, ...], ...

    def forward(self, x):
        self.outputs0 = [] #hooked intermediate features
        orig_result = self.model(x) #straightfoward outputs
        
        self.outputs = [None]*len(self.outputs0) #hooked intermediate features average
        
        N = x.shape[0] #mini-batch size
        M = len(self.outputs0)//N #number of resolutions
        
        for i in range(len(self.outputs0)): #pooling 2D features to feature response
            self.outputs[i] = nn.AvgPool2d(self.outputs0[i].shape[-1])(self.outputs0[i]).reshape(-1).to(device=device)
        #print ([self.outputs[i].shape for i in range(len(self.outputs))])
            
        self.features = [[None]*N for _ in range(M)] #M*N features
        for j in range(N):#j-th image 
            fpn_features = self.outputs[j:len(self.outputs):N]
            for i in range(len(fpn_features)): #i-th resolution
                self.features[i][j] = fpn_features[i]
                
        fpn_results = [self.model.fpn_classifiers[i](torch.stack(self.features[i])) for i in range(M)] #M layers feature -> score
        
        if model_choice != -2:
            all_results = torch.cat((orig_result, torch.cat(fpn_results,axis=1)), axis=1) #concatenate original decisions and intermediate decisions
        else:
            all_results = torch.cat(fpn_results,axis=1) #concatenate intermediate decisions
        all_results = nn.ReLU()(all_results) #remove noise
        final_output = self.ensemble(all_results) #final decision
        
        return final_output
        
        
def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        if step<10:
            print(imgs.mean())
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all
    
        
def do_main(epoch_indices=range(5,10), tta=3):
    # for training only, need nightly build pytorch
    global device

    seed_everything(CFG['seed'])
    
    #folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), train.label.values)
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    
    N = 4280 #val image number
    #N = len(list(os.listdir('../input/cassava-leaf-disease-classification/train_images')))
    Val_preds = np.zeros((len(CFG['used_epochs']),N,CFG['n_class']))
    



    device = torch.device(CFG['device'])
    model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique()).to(device)
    
    for fold, (trn_idx, val_idx) in enumerate(folds):
    #for fold in range(len(CFG['used_epochs'])):
        # we'll train fold 0 first
        if fold > 0:
            continue 

        print('Inference fold {} started'.format(fold))

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        #valid_ = train 
        valid_ds = CassavaDataset(valid_, '../input/cassava-leaf-disease-classification/train_images', transforms=get_inference_transforms(), output_label=False)
        
        val_loader = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )
        
        
        val_preds = []
        #tst_preds = []
        
        for i, epoch in enumerate(epoch_indices[fold]):    
            try:
                model.load_state_dict(torch.load('../archive/resnet50_fpn1_temperloss_local_lookahead/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)))
            except:
                continue
        
                
            #model.model.conv_head.weight[:,:,0,0] = torch.from_numpy(gaussian_filter(model.model.conv_head.weight[:,:,0,0].cpu().detach().numpy(), sigma=0.3))
            
            with torch.no_grad():
                for _ in range(tta):
                    val_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, val_loader, device)]
                    #tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*inference_one_epoch(model, tst_loader, device)]

        if len(CFG['used_epochs'][fold])>0:
            val_preds = np.mean(val_preds, axis=0)
        else:
            val_preds = np.zeros((N,CFG['n_class']))
            
        Val_preds[fold,...] = val_preds.copy()
        
        #del model
        #torch.cuda.empty_cache()
        
    ## val_preds is Nx5 matrix, Val_preds is 5xNx5 matrix
    Val_preds = np.mean(Val_preds, axis=0)
    Val_preds = Val_preds/np.sum(Val_preds,axis=1).reshape(-1,1)
    transformer = np.eye(5) #initialization
    ## modify 5x5 matrix:
    #transformer[0][0] = 0.95  
        
    #map test set scores to test set scores:
    Val_preds = np.dot(Val_preds , transformer)
        
        
    print('fold {} validation loss = {:.5f}'.format(fold, log_loss(valid_.label.values, Val_preds)))
    print('fold {} validation accuracy = {:.5f}'.format(fold, (valid_.label.values==np.argmax(Val_preds, axis=1)).mean()))
    print(valid_.label.values,np.argmax(Val_preds, axis=1))

        
if __name__ == '__main__':
    
    #for epoch_indices in range(20): #test single epoch
    #    try:
    #        do_main([epoch_indices],1)
    #    except:
    #        pass
        
    do_main(CFG['used_epochs'],CFG['tta']) #test all epochs with tta
