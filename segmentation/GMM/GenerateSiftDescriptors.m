function [siftArr] = GenerateSiftDescriptors( I, gridSpacing, patchSize, b_resize )

initialization
if size(I,3)==3,
    hsv = rgb2hsv(I);Iv=hsv(:,:,3); Is=hsv(:,:,2);Ih=hsv(:,:,1);
    %Iv=rgb2gray(I/255);
else Iv=double(I);
end

if b_resize,
	Imagesizeratio=sqrt(Imagesize(1)*Imagesize(2)/size(Iv,1)/size(Iv,2));
	Iv=resample(resample(Iv,round(Imagesizeratio*size(Iv,1)),size(Iv,1))',round(Imagesizeratio*size(Iv,2)),size(Iv,2))';
end

I=Iv;
[hgt wid] = size(I);


    %% make grid (coordinates of upper left patch corners)
    remX = mod(wid-patchSize,gridSpacing);
    offsetX = floor(remX/2)+1;
    remY = mod(hgt-patchSize,gridSpacing);
    offsetY = floor(remY/2)+1;
    
    [gridX,gridY] = meshgrid(offsetX:gridSpacing:wid-patchSize+1, offsetY:gridSpacing:hgt-patchSize+1);



    %% find SIFT descriptors
    siftArr = sp_find_sift_grid(I, gridX, gridY, patchSize, 0.8);
    siftArr = sp_normalize_sift(double(siftArr));
