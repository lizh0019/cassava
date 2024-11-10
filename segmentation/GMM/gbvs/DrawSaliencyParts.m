% img = imread(sprintf('samplepics/%d.jpg',i));
img = imread('D:\CQ\Saliency Detection\gbvs\demo\samplepics\1.jpg');
%   tic; 

    % this is how you call gbvs
    % leaving out params reset them to all default values (from
    % algsrc/makeGBVSParams.m)
    outImg = gbvs( img );   

%   toc;
  %outW = 320;
  % show result in a pretty way  
 
  %s = outW / size(img,2) ; 
  sz = size(img); sz = sz(1:2);
  %sz = round( sz * s );

  % img = imresize( img , sz , 'bicubic' );  
  saliency_map = imresize( outImg.master_map , sz , 'bicubic' );
  if ( max(img(:)) > 2 ) img = double(img) / 255; end
  img_thresholded = img .* repmat( saliency_map >= prctile(saliency_map(:),75) , [ 1 1 size(img,3) ] );  
  
  figure;
  subplot(2,2,1);
  imshow(img);
  title('original image');
  
  subplot(2,2,2);
  imshow(saliency_map);
  title('GBVS map');
  
  subplot(2,2,3);
  imshow(img_thresholded);
  title('most salient (75%ile) parts');

  subplot(2,2,4);
  show_imgnmap(img,outImg);
  title('saliency map overlayed');
  
  imwrite(img_thresholded, 'TA36a.jpg', 'jpg');
%   
%   if ( i < 5 )
%     fprintf(1,'Now waiting for user to press enter...\n');
%     pause;
%   end