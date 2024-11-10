function fasttrain(root,dataset)
initialization
load Hist_train

Pathname=strcat(root,dataset,'\');%'D:\databases\101_ObjectCategories\'
Category=dir(strcat(Pathname,'*.*'));%read the database


load offset

yapp = zeros(offset(end),1);
for i=1:CategoryNum-2,
    pos=offset(i+1);
    yapp(offset(i)+1:pos) = i; 
end

fid=fopen('Sparse_Hist_Train.txt', 'wt');
for i=1:size(yapp),
    fprintf(fid, '%d ', yapp(i));
    for j=1:size(Hist,2),
        if Hist(i,j)~=0,
            fprintf(fid, '%s %f ',strcat(num2str(j),':'),Hist(i,j));
        end
    end
    fprintf(fid, '\n');
end
fclose(fid)
!train.exe -s 5 -t 2 -c 10000 -g -1 -m 80 Sparse_Hist_Train.txt CVM.model.txt
!predict.exe Sparse_Hist_Train.txt CVM.model.txt CVM.output.txt