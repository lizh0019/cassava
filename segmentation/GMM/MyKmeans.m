function MyKmeans(root)

initialization

dataset='\Train';
Pathname=strcat(root,dataset,'\');%'D:\databases\101_ObjectCategories\'
Category=dir(strcat(Pathname,'*.*'));%read the database
CategoryNum=size(Category,1);%number of categories
%Ensemble=zeros(Imagesize(1)*Imagesize(2)/8/8*100*15,128);
Ensemble=zeros(Ensemblesize,feature_dim,'single');
%Ensemble=zeros(1e+4,256,'single');
%des_num_per_image=floor(size(Ensemble,1)/25/140)
% temp=randn(50,50);temp=temp-mean(mean(temp));
% temp=temp/std(reshape(temp,1,50*50));
% Q=reshape(abs(dct2(temp)),1,50*50);
% save Q Q
 load Q_15s
C=[];
imagenum=0;  
%% feature extraction
for Ii=1:CategoryNum,

    if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 

        foldername=Category(Ii).name;
        Image=dir(strcat(Pathname,foldername,'\*.jpg'));
        %ensemble=single(zeros(Imagesize(1)*Imagesize(2)/8/8*length(Image),128));
        for k=1:length(Image),
            %I = imread(strcat(Pathname,foldername,'\',Image(k).name));Image(Ii).size=size(I);I=double(I);
    
% entropyfile1=strcat(Pathname,foldername,'\',Image(k).name,'.sifte');
% entropyfile2=strcat(Pathname,foldername,'\',Image(k).name,'.klentropy1');
% fr1 = fopen(entropyfile1, 'r');
% fr2 = fopen(entropyfile2, 'r');
% descriptors1=fread(fr1,'double');
% descriptors1=reshape(descriptors1,length(descriptors1)/128,128);
% descriptors2=fread(fr2,'double');
% descriptors2=reshape(descriptors2,length(descriptors2)/128,128);
% fclose(fr1);
% fclose(fr2);
% descriptorssize=min(size(descriptors1,1),size(descriptors2,1));
% descriptors=[descriptors1(1:descriptorssize,:),descriptors2(1:descriptorssize,:)];   
% descriptors=descriptors(:,1:128);
               entropyfile=strcat(Pathname,foldername,'\',Image(k).name,filetype);%sift2');
               fr = fopen(entropyfile, 'r');

                    %descriptors=fread(fr,floor(size(I,1)/8-1)*floor(size(I,2)/8-1)*128,'double');
                    descriptors=fread(fr,'double');
                    descriptors=reshape(descriptors,length(descriptors)/feature_dim,feature_dim);
                    fclose(fr);
                des_num_per_image=floor(size(Ensemble,1)/(CategoryNum-2)/length(Image));
                patchsize=min(des_num_per_image,size(descriptors,1));
                if ~(strcmp(filetype,'.dasfasd'))
                    patchindex=randperm(size(descriptors,1));
                    descriptors=descriptors(patchindex(1:patchsize),:);
                    Ensemble(imagenum+1:imagenum+patchsize,:)=descriptors;
                else
%                 step=floor(size(descriptors,1)/patchsize);
%                 tmpdescriptors=[];
%                 for t=1:patchsize,
%                     tmpdescriptors=[tmpdescriptors,descriptors((t-1)*step+1:t*step,:)];
%                 end
%                 tmpdescriptors=mean(tmpdescriptors);
%                 tmpdescriptors=reshape(tmpdescriptors',128,patchsize);
%                 Ensemble(imagenum+1:imagenum+patchsize,:)=tmpdescriptors';
                
% descriptorsMean=mean(descriptors);
% descriptorsStd=std(descriptors);
% descriptors=(descriptors-repmat(descriptorsMean,size(descriptors,1),1))./repmat(descriptorsStd,size(descriptors,1),1);

                %[IDX,C0] = kmeans(descriptors,patchsize,'distance','cityblock','replicates',1,'emptyaction','singleton','start','cluster','MaxIter',10);
                [C0, IDX]=vl_kmeans(descriptors', patchsize,  'distance', 'l1', 'algorithm', 'elkan') ;C0=C0';
                C=NNmapping(C0,descriptors,1);

 %C=double(C.*repmat(descriptorsStd,size(C,1),1)+repmat(descriptorsMean,size(C,1),1));
%                     Mindescriptors=min(min(descriptors));
%                     Maxdescriptors=max(max(descriptors));
%                     scale=255/(Maxdescriptors-Mindescriptors);
%                     descriptors=(descriptors-Mindescriptors)*scale;
%                     descriptors=uint8(descriptors);descriptors=descriptors';
%                     [tree] = vl_hikmeans(descriptors,patchsize,10,'method', 'elkan') ; 
%                     C=tree.centers;C=double(double(C')/scale+Mindescriptors);
                    
                    Ensemble(imagenum+1:imagenum+patchsize,:)=C;
                end
                
                imagenum=imagenum+patchsize;                
        end
        display(strcat(foldername,' feature loaded'))
    end


end

Ensemble=Ensemble(~~sum(Ensemble,2),:);

 save Ensemble Ensemble
% 
% load Ensemble
%save Ensemble.dat Ensemble -ascii

% codevectorsize=size(Ensemble,1)
% 
% %if ~(strcmp(filetype,'.klentropy2'))
%     EnsembleMean=mean(Ensemble);
% % else
% %     EnsembleMean=zeros(1,128);
% % end
% EnsembleStd=std(Ensemble);
% save EnsembleMeanVar EnsembleMean EnsembleStd
%  Ensemble=(Ensemble-repmat(EnsembleMean,size(Ensemble,1),1))./repmat(EnsembleStd,size(Ensemble,1),1);
% %Ensemble=(Ensemble-repmat(EnsembleMean,size(Ensemble,1),1));
% 
% %MinEnsemble=min(min(Ensemble));
% %MaxEnsemble=max(max(Ensemble));
% %scale=255/(MaxEnsemble-MinEnsemble)
% %EnsembleInt=(Ensemble-MinEnsemble)*scale;% to make it approx. 0~255
% display('begin clustering')
%  tic
% % [IDX,C] = kmeans(Ensemble,codewordnum,'distance','sqEuclidean','replicates',1,'emptyaction','singleton','start','uniform');
% % C=double(double(C)/scale+MinEnsemble);
% 
% 
% %EnsembleInt=uint8(EnsembleInt);EnsembleInt=EnsembleInt';
% % [tree] = vl_hikmeans(EnsembleInt,codewordnum,100,'method', 'elkan') ; % default
% % C=tree.centers;C0=double(double(C')/scale+MinEnsemble);% to reverse
% [C0, A, ENERGY]=vl_kmeans(Ensemble', codewordnum, 'verbose', 'distance', 'l1', 'algorithm', 'elkan') ;C0=C0';
% % if (strcmp(filetype,'.klentropy2')  || strcmp(filetype,'.sift2'))
% %      [IDX,C0] = kmeans(Ensemble,codewordnum,'distance','cityblock','replicates',1,'emptyaction','singleton','start','sample','MaxIter',1000);
% % else 
% %      [IDX,C0] = kmeans(Ensemble,codewordnum,'distance','sqEuclidean','replicates',1,'emptyaction','singleton','start','sample','MaxIter',1000);
% % end
% C=double(C0);
% %C=NNmapping(C0,Ensemble,1);
% 
%  %C=double(C.*repmat(EnsembleStd,codewordnum,1)+repmat(EnsembleMean,codewordnum,1));
% 
% display('end clustering, minutes:')
% toc/60
% save codebook_GlobalSIFT.txt C -ascii