function BoW_GlobalSIFT2(root,dataset)
initialization
%% obtain histograms for each image

load mytree
Pathname=strcat(root,dataset,'\');
Category=dir(strcat(Pathname));
CategoryNum=size(Category,1);

load EnsembleMeanVar 

if ~isempty(strfind(lower(dataset),'test')),
    Hist=zeros(MaxNumImage,codewordnum,'single');
end
if ~isempty(strfind(lower(dataset),'train')),
    Hist=zeros(MaxNumImage,codewordnum,'single');
end


offset=[0];
ImageName={};
for Ii=1:CategoryNum,
    if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 
        foldername=Category(Ii).name;
        Image=dir(strcat(Pathname,foldername,'\*.jpg'));
        imgnum=0;
        for k=1:length(Image),

               entropyfile=strcat(Pathname,foldername,'\',Image(k).name);dotpos=find(entropyfile=='.');
               entropyfile=strcat(entropyfile(1:dotpos(end)-1),'.mat');%feature file is *.mat
               load(entropyfile);%descriptors=descriptors*310;
               
               des_weight=abs(des_weight).^0.5;%descriptor weight normalization
if (BoW)
            descriptorspath= vl_ikmeanspush(uint8(descriptors'),int32(mytree'));
            hist_test=hist(descriptorspath,[1:codewordnum]);
end
if (SVT)
            descriptorspath= vl_hikmeanspush(mytree,uint8(descriptors'));
            hist_test = single(vl_hikmeanshist2(mytree,descriptorspath,des_weight));
end
            imgnum=imgnum+1;
            Hist(offset(end)+imgnum,:)=hist_test;%histogram for every image
        end
        offset=[offset,offset(end)+imgnum];
    end
    display(strcat(foldername,' histogram calculated'))
end
yapp = zeros(offset(end),1);
for i=1:CategoryNum-2,
    pos=offset(i+1);
    yapp(offset(i)+1:pos) = i; 
end
categoryindex = yapp;

totalimgnum=offset(end);
if ~isempty(strfind(lower(dataset),'test')),
    testoffset=offset;
    save Hist_test Hist CategoryNum totalimgnum testoffset categoryindex ImageName
end
if ~isempty(strfind(lower(dataset),'train')),
    trainoffset=offset;
    save Hist_train Hist CategoryNum totalimgnum trainoffset categoryindex ImageName
end
