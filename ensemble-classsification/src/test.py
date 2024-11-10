package_path = '../input/timm-pytorch-image-models/pytorch-image-models-master' #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
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

model_choice = 1 #0, 1, 2, 3
CFG = {
    'fold_num': 5,
    'n_class': 5,
    'seed': 719,
    'model_arch': ['tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b6_ap'][model_choice],
    'test_bs': 32, #6, #32
    'num_workers': 4,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [5, 6,7, 8, 9], #range(10), #[6,7,8,9],
    'weights': [1]*10
}
fpn_dim = {'tf_efficientnet_b4_ns':[24,32,56,112,160,272,448], 'tf_efficientnet_b5_ap':[24,40,64,128,176,304,512], 'tf_efficientnet_b6_ap':[32,40,72,144,200,344,576], 'tf_efficientnet_b7_ap':[32,48,80,160,224,384,640]}
resolution = {'tf_efficientnet_b4_ns':380, 'tf_efficientnet_b5_ap':456, 'tf_efficientnet_b6_ap':528, 'tf_efficientnet_b7_ap':600}
#resolution = {'tf_efficientnet_b4_ns':528, 'tf_efficientnet_b5_ap':528, 'tf_efficientnet_b6_ap':528, 'tf_efficientnet_b7_ap':528}
CFG['img_size_width'] = CFG['img_size_height'] = resolution[CFG['model_arch']]

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



def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size_height'], CFG['img_size_width']),
            Transpose(p=0.5),
            CoarseDropout(p=0.5),
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
        self.model = timm.create_model(model_arch, pretrained=pretrained) #pretrained model
        self.model.classifier = nn.Linear(self.model.classifier.in_features, n_class) #original decision
        
        self.fpn_sizes = fpn_dim[CFG['model_arch']] #each layer feature depth
        self.model.fpn_classifiers = [nn.Linear(self.fpn_sizes[i], n_class).to(device) for i in range(len(self.fpn_sizes))] #seperate decisions
        self.ensemble = nn.Linear((1+len(self.fpn_sizes))*n_class, n_class) #ensemble all layers
        self.outputs = [] #hooked intermediate features
        for name, layer in self.model.blocks._modules.items():
            layer.register_forward_hook(self.myhook) #record intermediate features
        
    def myhook(self, module, input, output):
        self.outputs.extend(output) #0-th layer:[0-th image, 1-th, ...], 1-th layer:[0-th image, 1-th, ...], ...

    def forward(self, x):
        self.outputs = [] #hooked intermediate features
        orig_result = self.model(x) #straightfoward outputs
        
        N = x.shape[0] #mini-batch size
        M = len(self.outputs)//N #number of resolutions
        
        for i in range(len(self.outputs)): #pooling 2D features to feature response
            self.outputs[i] = nn.AvgPool2d(self.outputs[i].shape[-1])(self.outputs[i]).reshape(-1).to(device=device)
            
        self.features = [[None]*N for _ in range(M)] #M*N features
        for j in range(N):#j-th image 
            fpn_features = self.outputs[j:len(self.outputs):N]
            for i in range(len(fpn_features)): #i-th resolution
                self.features[i][j] = fpn_features[i]
                
        fpn_results = [self.model.fpn_classifiers[i](torch.stack(self.features[i])) for i in range(M)] #M layers feature -> score
        all_results = torch.cat((orig_result, torch.cat(fpn_results,axis=1)), axis=1) #concatenate original decisions and intermediate decisions
        all_results = nn.ReLU()(all_results) #remove noise
        final_output = self.ensemble(all_results) #final decision
        
        return final_output
        
        
def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all
    
        
if __name__ == '__main__':
     # for training only, need nightly build pytorch

    seed_everything(CFG['seed'])
    
    
    for fold in range(CFG['fold_num']):
        # we'll test fold 0 first
        if fold > 0:
            break 

        print('Inference fold {} started'.format(fold))

        test = pd.DataFrame()
        test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
        test_ds = CassavaDataset(test, '../input/cassava-leaf-disease-classification/test_images/', transforms=get_inference_transforms(), output_label=False)

        tst_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=CFG['test_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])
        model = CassvaImgClassifier(CFG['model_arch'], CFG['n_class']).to(device)
        
        tst_preds = []
        
        for i, epoch in enumerate(CFG['used_epochs']):
            model_name = '../input/best_model/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch)#.replace('_','-')
            print (model_name)
            #if os.path.exists('model_name'):
            #    print (model_name)
            model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))
            #model = model.to(device)
            #else:
            #    continue
            
            with torch.no_grad():
                for _ in range(CFG['tta']):
                    image_preds_all = inference_one_epoch(model, tst_loader, device)
                    tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*image_preds_all]

        tst_preds = np.mean(tst_preds, axis=0) 
        
        ## tst_preds is Nx5 matrix, do softmax
        #tst_preds=nn.Softmax(dim=1)(torch.from_numpy(tst_preds))
        tst_preds = tst_preds/np.sum(tst_preds,axis=1).reshape(-1,1)
        #transformer = torch.FloatTensor(np.eye(5)) #initialization
        transformer = np.eye(5) #initialization
        ## modify 5x5 matrix:
        #transformer[0][0] = 0.95  
        
        #map test set scores to test set scores:
        #tst_preds = torch.matmul(tst_preds, transformer).numpy()
        tst_preds = np.dot(tst_preds , transformer)
        
        del model
        torch.cuda.empty_cache()
        
test['label'] = np.argmax(tst_preds, axis=1)
print (test.head())

test.to_csv('submission.csv', index=False)
