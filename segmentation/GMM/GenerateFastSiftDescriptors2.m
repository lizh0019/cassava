function [descriptors, des_weight] = GenerateFastSiftDescriptors2( I, step, patchsize, b_resize )
%function [] = GenerateSiftDescriptors( imageFileList, imageBaseDir, dataBaseDir, maxImageSize, gridSpacing, patchSize, canSkip )
%
%Generate the dense grid of sift descriptors for each image
%
% imageFileList: cell of file paths
% imageBaseDir: the base directory for the image files
% dataBaseDir: the base directory for the data files that are generated
%  by the algorithm. If this dir is the same as imageBaseDir the files
%  will be generated in the same location as the image files
% maxImageSize: the max image size. If the image is larger it will be
%  resampeled.
% gridSpacing: the spacing for the grid to be used when generating the
%  sift descriptors
% patchSize: the patch size used for generating the sift descriptor
% canSkip: if true the calculation will be skipped if the appropriate data 
%  file is found in dataBaseDir. This is very useful if you just want to
%  update some of the data or if you've added new images.



initialization
if size(I,3)==3,
    while size(I,1)*size(I,2)>2000*2000,
        I=I(1:2:end,1:2:end,:);
    end
    %hsv = rgb2hsv(I);Iv=hsv(:,:,3); Is=hsv(:,:,2);Ih=hsv(:,:,1);
    Iv=rgb2gray(I/255);
else Iv=double(I);
end

if b_resize,
	Imagesizeratio=sqrt(Imagesize(1)*Imagesize(2)/size(Iv,1)/size(Iv,2));
	Iv=resample(resample(Iv,round(Imagesizeratio*size(Iv,1)),size(Iv,1),2)',round(Imagesizeratio*size(Iv,2)),size(Iv,2),2)';
%     Imagesizeratio1=Imagesize(1)/size(Iv,1);Imagesizeratio2=Imagesize(2)/size(Iv,2);
%     Iv=resample(resample(Iv,round(Imagesizeratio1*size(Iv,1)),size(Iv,1))',round(Imagesizeratio2*size(Iv,2)),size(Iv,2))';
end


    %% make grid (coordinates of upper left patch corners)

% step=8;
% temp1=repmat([(step-1):step:(size(Iv,2)-step)]',1,round(size(Iv,1)/step-1));temp2=reshape(temp1',size(temp1,1)*size(temp1,2),1)';
% temp3=[(step-1):step:(size(Iv,1)-step)];temp4=repmat(temp3,1,ceil(length(temp2)/length(temp3)));
% fc=[temp2;temp4(1:length(temp2))];



    %% find SIFT descriptors
% fc1=[fc;4*ones(1,size(fc,2));zeros(1,size(fc,2))];[frames, descriptors] = vl_sift(single(Iv),'magnif',4,'frames',fc1) ;descriptors=double(descriptors');
%  binSize = 8 ;
%  magnif = 3 ;
%  Iv = vl_imsmooth(Iv, sqrt((binSize/magnif)^2 - .25)) ;
[frames, descriptors] = vl_dsift(single(Iv),'step',step,'size',patchsize/4) ;descriptors=double(descriptors');
%% FOR REFERENCE : descriptors=GenerateFastSiftDescriptors( I, 8, 16, 1 )%% -->( I, step, patchsize, b_resize )
%[frames, descriptors] = vl_sift(single(Iv),'magnif',4) ;descriptors=double(descriptors');

    %% Find Saliency Mapping

outImg = gbvs(Iv);   
sz = size(Iv); sz = sz(1:2);
saliency_map = imresize( outImg.master_map , sz , 'bicubic' );


%           x
%  ---------------------
%  |                   |
%  |                   | y
%  |                   |
%  |                   |
%  ---------------------


[y1 x1] = size(frames); 
[y2 x2] = size(saliency_map);
des_weight = zeros((y1+1),x1); 
weight = zeros((y1+1),x1); %test variable
check = 1;
                                   
for i= 1:x1  
    des_weight(:,i) = [frames(1,i);frames(2,i);saliency_map(frames(2,i),frames(1,i))]; %store X axis; Y axis; Weight
    %weight(:,i) = [frames(1,i);frames(2,i);saliency_map(frames(2,i),frames(1,i))];                      
end

