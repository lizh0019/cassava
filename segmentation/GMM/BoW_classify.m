function BoW_classify(root,dataset,K)
%K loose of result
%% classify
initialization
load NET
load Hist_test
CategoryNum=size(net,1);
Pathname=strcat(root,dataset,'\');%'D:\databases\101_ObjectCategories\'
Category=dir(strcat(Pathname,'*.*'));%read the database

ClassifierResult=fopen('D:\content_context\BoWBaseline_entropy\FSVMresult.txt','wt');
imgnum=0;
accuracy=0;
% %figure,row=floor(size(Hist,1)^0.5);column=floor(size(Hist,1)/row)+1;
% for Ii=1:CategoryNum+2,
%     if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 
%         foldername=Category(Ii).name;
%         Image=dir(strcat(Pathname,foldername,'\*.jpg'));
%         for k=1:length(Image),
%             enquiry=strcat(Pathname,foldername,'\',Image(k).name)
%             fprintf(ClassifierResult, '%s\n', enquiry);
%             score=zeros(1,CategoryNum-2);
%             imgnum=imgnum+1;
%             for i=1:CategoryNum,
%                 [Y, Y1] = svmfwd(net(i), Hist(imgnum,:));
%                 score(i)=Y1;
%             end
% 
%             [result,results]= classmatch(Category,foldername,score,K);%result:right or wrong;results:keywords
%             results
%             accuracy=accuracy+result;
%             Result=[];
%             for kk=K:-1:1,
%                 Result=strcat(Result,results(kk).name,', ');
%             end
%             fprintf(ClassifierResult, strcat(Result,'\n'));
%             %I=imread(enquiry);
%             %subplot(row,column,imgnum),subimage(I);title(Result),axis off
%             
%         end
%     end
% end
% accuracy/imgnum
cd 'D:\program_library\SVM\'
CategoryNum=net.CategoryNum;
[ypred] = svmmultival(Hist,net.xsup,net.w,net.b,net.nbsv,net.kernel,net.kerneloption);
cd 'D:\content_context\BoWBaseline_entropy'


%figure,row=floor(size(Hist,1)^0.5);column=floor(size(Hist,1)/row)+1;
Confusion.matrix=zeros(CategoryNum,CategoryNum);
Confusion.name={};
for Ii=1:CategoryNum+2,
    if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 
        foldername=Category(Ii).name;Confusion.name{Ii-2}=foldername;
        Image=dir(strcat(Pathname,foldername,'\*.jpg'));
        for k=1:length(Image),
            enquiry=strcat(Pathname,foldername,'\',Image(k).name);
            fprintf(ClassifierResult, '%s\n', enquiry);
            imgnum=imgnum+1;
            Confusion.matrix(Ii-2,ypred(imgnum))=Confusion.matrix(Ii-2,ypred(imgnum))+1;
            Result=Category(ypred(imgnum)+2).name;
            result=strcmp(foldername,Result);
            accuracy=accuracy+result;
            fprintf(ClassifierResult, strcat(Result,'\n'));    
            %I=imread(enquiry);subplot(row,column,imgnum),subimage(I);title(Result),axis off
        end
    end
end
save Confusion Confusion
Matrix=Confusion.matrix;
matrix=Matrix./repmat(sum(Matrix,2),1,CategoryNum);
diagmatrix=diag(matrix);
[score,index]=sort(diagmatrix,'descend');
for i=1:CategoryNum,Confusion.name{index(i)},score(i),end
accuracy0=accuracy/imgnum
accuracy1=mean(diagmatrix(1:CategoryNum))

%load Confusion;for i=1:size(Confusion.matrix,1),retrieval=strcmp(Confusion.name{i},'scissors');if(1==retrieval),Confusion.name{i},Confusion.matrix(i,i)/sum(Confusion.matrix(i,:)),end,end












