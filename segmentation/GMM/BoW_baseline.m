%% initialization
clear,clc,close all,fclose all

% cd '.\toolbox\';
% vl_setup;
% cd '..';

initialization

%% choose the database path
%root='D:\databases\Caltech 3';
%root='D:\databases\bad 4';
root='D:\databases\good&bad';
%root='D:\databases\NTU scene';
%root='D:\databases\landmark';
%root='D:\databases\event_dataset';
%root='D:\databases\sigapore picture';
%root='D:\databases\scene_categories';
%root='D:\databases\Caltech 101';
%root='D:\databases\NTU-25';
%root='D:\databases\scene 15';
%root='D:\content_context\BowBaseline\NTU scene';
%TrainTestPartition('D:\databases\Caltech 2\');
root='/home/lizhen/study/mypapers/under\ work/object\ localization/code/objects';

%% compute the feature vectors for codebook generation 

tic,dataset='\train';BoW_feature(root,dataset),toc
tic,dataset='\test';BoW_feature(root,dataset),toc
%% do clustering to generate codebook
% tic,
% MyKmeans(root);
% toc

%% compute the histograms of training images
dataset='\train';tic,BoW_GlobalSIFT(root,dataset),toc

%% train the classifiers
dataset='\train';tic,fasttrain(root,dataset),toc

%% compute the histograms of test images
dataset='\test';tic,BoW_GlobalSIFT(root,dataset),toc

%% classify test images
dataset='\test';tic,fastclassify(root,dataset),classificationtime=toc




