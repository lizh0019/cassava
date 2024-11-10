package_paths = [
    '../input/pytorch-image-models/pytorch-image-models-master', #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
    '../input/image-fmix/FMix-master'
]
import sys; 

for pth in package_paths:
    sys.path.append(pth)
    
from fmix import sample_mask, make_low_freq_image, binarise_mask
from bi_tempered_loss import *
t1, t2 = 0.2, 1.0
label_smoothing = 0.3
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
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import timm

from catalyst.data.sampler import BalanceClassSampler
import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import cv2
#import pydicom
#from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom
from lookahead import Lookahead


model_choice = 0 #-2, -1, 0, 1, 2, 3
CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': ['tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b7_ap', 'resnet50', 'tf_efficientnet_b3_ns'][model_choice],
    #'img_size_width': 380, #456, #380, #int(800*0.9), #512, #600, #512,528, 
    #'img_size_height': 380, #456, #380, #int(600*0.9), #512, #600, #600,528, 
    'epochs': 10,
    'train_bs': [8, 6, 4, 1, 16, 32][model_choice],
    'valid_bs': [8, 6, 4, 1, 16, 32][model_choice],
    'T_0': 10,
    'lr': 2e-4, #1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}
fpn_dim = {'tf_efficientnet_b4_ns':[24,32,56,112,160,272,448], 'tf_efficientnet_b5_ap':[24,40,64,128,176,304,512], 'tf_efficientnet_b6_ap':[32,40,72,144,200,344,576], 'tf_efficientnet_b7_ap':[32,48,80,160,224,384,640], 'resnet50':[256,256,256,512,512,512,512,1024,1024,1024,1024,1024,1024,2048,2048,2048], 'tf_efficientnet_b3_ns':[24,32,48,96,136,232,384]}
resolution = {'tf_efficientnet_b4_ns':512, 'tf_efficientnet_b5_ap':468, 'tf_efficientnet_b6_ap':528, 'tf_efficientnet_b7_ap':600, 'resnet50': 512, 'tf_efficientnet_b3_ns':300}
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

img = get_img('../input/cassava-leaf-disease-classification/train_images/336550.jpg')
#plt.imshow(img)
#plt.show()
print (img.shape)
##dataset

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (CFG['img_size_width'], CFG['img_size_height']),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
        
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                #print(mask.shape)

                #assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum()/CFG['img_size_width']/CFG['img_size_height']
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False
        
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((CFG['img_size_width'], CFG['img_size_height']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_size_width'] * CFG['img_size_height']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img
            
            
##Define Train\Validation Image Augmentations
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    if CFG['img_size_width'] == CFG['img_size_height']:
        return Compose([
            RandomResizedCrop(CFG['img_size_width'], CFG['img_size_height']),
            Transpose(p=0.5), #only applicable when image_size width==height
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
    else:
        return Compose([
            RandomResizedCrop(CFG['img_size_width'], CFG['img_size_height']),
            #Transpose(p=0.5), #only applicable when image_size width==height
            CoarseDropout(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['img_size_height'], CFG['img_size_width'], p=1.),
            Resize(CFG['img_size_height'], CFG['img_size_width']),
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
        
        
def prepare_dataloader(df, trn_idx, val_idx, data_root='../input/cassava-leaf-disease-classification/train_images/'):
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
        
    train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
    valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=CFG['num_workers'],
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        #print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)   #output = model(input)
            #print(image_preds.shape, exam_pred.shape)

            #loss = loss_fn(image_preds, image_labels)
            losses = [bi_tempered_logistic_loss(image_preds[i], torch.FloatTensor([0]*image_labels[i]+[1]+[0]*(image_preds.shape[1]-image_labels[i]-1)).to(device), t1=t1, t2=t2, label_smoothing=label_smoothing) for i in range(image_preds.shape[0])]
            loss = torch.mean(torch.stack(losses)) 
            
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        #loss = loss_fn(image_preds, image_labels)
        losses = [bi_tempered_logistic_loss(image_preds[i], torch.FloatTensor([0]*image_labels[i]+[1]+[0]*(image_preds.shape[1]-image_labels[i]-1)).to(device), t1=t1, t2=t2, label_smoothing=label_smoothing) for i in range(image_preds.shape[0])]
        loss = torch.mean(torch.stack(losses))
        
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()
            
            
# reference: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
class MyCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
        
        
        
if __name__ == '__main__':
     # for training only, need nightly build pytorch

    seed_everything(CFG['seed'])
    
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)
    
    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        if fold == 0:
            continue 

        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root='../input/cassava-leaf-disease-classification/train_images/')

        device = torch.device(CFG['device'])
        
        model_name = '../results/{}_fold_{}'.format(CFG['model_arch'], fold)
        if os.path.exists(model_name):
            model_dict = torch.load(model_name)
            model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
            model.load_state_dict(model_dict)
            model = model.to(device=device)
            print ('resumed training')
        else:
            model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
            print ('train from scratch')
            
        scaler = GradScaler()   
        #optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        optimizer = Lookahead(torch.optim.Adam(model.parameters(), lr=CFG['lr'])) #, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.3, amsgrad=True))#, k=5, alpha=0.5) # Initialize Lookahead
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
        #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))
        
        loss_tr = nn.CrossEntropyLoss().to(device) #MyCrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        
        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)

            model_name = '../results/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)
            torch.save(model.state_dict(),model_name)
            
            with torch.no_grad():
                valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
            
            
        #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()
        
        
        
