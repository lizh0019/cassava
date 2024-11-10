%% initialization


foldername=[];filename=[];%category name and image name
Ensemble=[];%aggragate trainvector
Imagesize=[320 240];%downsize image if larger than this size

offset=[0];%index offset of each category 

feature_dim=128;%dimension of feature in each patch
Ensemblesize=1.5e+4;
codewordnum=400;%size of codebook

subpatchsize=48;%9,36,48;
MaxImagePatch=3;

quantizelevel=[128];
stdskip=5;
filetype='.sift_sep';%klentropy  sift  sifte siftentropy

