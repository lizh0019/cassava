function fastclassify(root,dataset)
%% classify
initialization

load Hist_test

Pathname=strcat(root,dataset,'\');%'D:\databases\101_ObjectCategories\'
Category=dir(strcat(Pathname,'*.*'));%read the database
CategoryNum=size(Category,1);
offset=[0];


for Ii=1:CategoryNum,
    if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 
        foldername=Category(Ii).name;
        Image=dir(strcat(Pathname,foldername,'\*.jpg'));
        imgnum=0;
        for k=1:length(Image),
               entropyfile=strcat(Pathname,foldername,'\',Image(k).name,filetype);
               fr = fopen(entropyfile, 'r');
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

                fid=fopen('Sparse_Hist_Test.txt', 'wt');
                for i=1:size(descriptors,1),
                    fprintf(fid, '%d ', Ii-2);
                    for j=1:size(descriptors,2),
                        if descriptors(i,j)~=0,
                            fprintf(fid, '%s %f ',strcat(num2str(j),':'),descriptors(i,j));
                        end
                    end
                    fprintf(fid, '\n');
                end
                fclose(fid);

                !predict.exe Sparse_Hist_Test.txt CVM.model.txt CVM.output.txt


                fid = fopen('CVM.output.txt', 'r');
                ypred= fscanf(fid,'%d');
                fclose(fid);
                I=imread(strcat(Pathname,foldername,'\',Image(k).name));
                ypred=reshape(ypred,floor(size(I,1)/8),floor(size(I,2)/8));
                figure,imshow(I,[]);
                figure,imshow(ypred,[]);


        end
        offset=[offset,offset(end)+imgnum];
    end
    display(strcat(foldername,' histogram calculated'))
end
