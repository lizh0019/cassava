function TrainTestPartition(Pathname)

%Pathname='D:\databases\Caltech 101\';
Category=dir(strcat(Pathname,'*.*'));%read the database
CategoryNum=size(Category,1);%number of categories

%% feature extraction
for Ii=1:CategoryNum,
    if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 
        foldername=Category(Ii).name;
        Image=dir(strcat(Pathname,foldername,'\*.pgm'));
        index=randperm(length(Image));
        mkdir(strcat(Pathname,'train\',foldername,'\'));
        for k=1:floor(length(Image)*3/10),
            I = imread(strcat(Pathname,foldername,'\',Image(index(k)).name));
            newname = strcat(Pathname,'train\',foldername,'\',Image(index(k)).name);
            newname = strcat(newname(1:end-4),'.jpg');
            movefile(strcat(Pathname,foldername,'\',Image(index(k)).name),newname,'f');

            imwrite(I,newname,'jpg');
        end
        mkdir(strcat(Pathname,'test\',foldername,'\'));
        for k=floor(length(Image)*3/10)+1:length(Image),
            I = imread(strcat(Pathname,foldername,'\',Image(index(k)).name));
            newname = strcat(Pathname,'test\',foldername,'\',Image(index(k)).name);
            newname = strcat(newname(1:end-4),'.jpg');
            movefile(strcat(Pathname,foldername,'\',Image(index(k)).name),newname,'f');

            imwrite(I,newname,'jpg');
        end
        rmdir(strcat(Pathname,foldername,'\'),'s')
        display(strcat(foldername,' partitioned'))
    end
end


