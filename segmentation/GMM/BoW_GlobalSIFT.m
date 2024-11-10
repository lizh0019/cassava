function BoW_GlobalSIFT(root,dataset)
initialization
%% obtain histograms for each image

Pathname=strcat(root,dataset,'\');%'D:\databases\101_ObjectCategories\'
Category=dir(strcat(Pathname,'*.*'));%read the database
CategoryNum=size(Category,1);

Hist=zeros(Ensemblesize,feature_dim);
offset=[0];



for Ii=1:CategoryNum,
    if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 
        foldername=Category(Ii).name;
        Image=dir(strcat(Pathname,foldername,'\*.jpg'));
        imgnum=0;
        for k=1:length(Image),



               entropyfile=strcat(Pathname,foldername,'\',Image(k).name,filetype);


               fr = fopen(entropyfile, 'r');
               %fr = -1;
               if fr == -1,
                    fw = fopen(entropyfile, 'w');
                    enquiry=strcat(Pathname,foldername,'\',Image(k).name);
                    Itest = imread(enquiry);Image(Ii).size=size(Itest);I=double(Itest);     

                    if strcmp(filetype,'.sift_sep'),
                        descriptors=GenerateSiftDescriptors( I, 8, 8, 1);
                    end

                    if strcmp(filetype,'.sift'),
                        descriptors=GenerateSiftDescriptors( I, 8, 16 ,1);descriptors = sp_normalize_sift(double(descriptors));
                    end

                    fwrite(fw, descriptors, 'double'); 
                    fclose(fw);
                else
                    descriptors=fread(fr,'double');
                    descriptors=reshape(descriptors,length(descriptors)/feature_dim,feature_dim);
                    fclose(fr);
               end
            
                    des_num_per_image=floor(size(Hist,1)/(CategoryNum-2)/length(Image));
                    patchsize=min(des_num_per_image,size(descriptors,1));
                    patchindex=randperm(size(descriptors,1));
                    patchindex=sort(patchindex(1:patchsize));
            Hist((offset(end)+imgnum+1):(offset(end)+imgnum+patchsize),:)=descriptors(patchindex,:);%histogram for every image
            imgnum=imgnum+patchsize;
        end
        offset=[offset,offset(end)+imgnum];
    end
    display(strcat(foldername,' histogram calculated'))
end

Hist=Hist(1:offset(end),:);

totalimgnum=offset(end);
if ~isempty(strfind(lower(dataset),'test')),
    save Hist_test Hist CategoryNum totalimgnum
end
if ~isempty(strfind(lower(dataset),'train')),
    save Hist_train Hist CategoryNum totalimgnum
    save offset offset
end
