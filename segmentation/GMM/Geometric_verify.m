function Geometric_verify0(root,dataset,b_siftmatch, b_ransac, b_selkdtree, topN)
initialization

load Hist_test 
% HistSum=single(sum(Hist,2));
%TBT = Hist(boolean(HistSum),:);
TBT=Hist;
timgindex = categoryindex;
timgnump = testoffset;
% testimgnum = totalimgnum; 
% testimgname = ImageName;
% [testimgname,testimgnum] = GetImageName(root, '\Test');
% save QueryName testimgname testimgnum
%load QueryName

load Hist_train
% HistSum=single(sum(Hist,2));
% TrainBT = Hist(boolean(HistSum),:);
TrainBT=Hist;

trainindex = categoryindex;
trainimgnump = trainoffset;
% trainimgname = ImageName;
% trainimgnum = totalimgnum; 
% [trainimgname,trainimgnum] = GetImageName(root, '\Train');
% save DatabaseName trainimgname trainimgnum
%load DatabaseName
clear Hist

NumCategory=max(categoryindex);
Retrieve_result = []; %% just added

load mytree

if b_selkdtree == 1
   TrainBT  = single (TrainBT (1:end,:)' );
   kdforest = vl_kdtreebuild(TrainBT,'NumTrees',20);
   save kdforest kdforest
   TrainBT = TrainBT';
end

count = 0; 
subrate = [];
subcount = 0;
recordcateg = 1;

TrainBTsum=single(sum(TrainBT,2)+eps);
TrainBT=bsxfun(@rdivide, TrainBT, TrainBTsum);
TBTsum=single(sum(TBT,2)+eps);
TBT=bsxfun(@rdivide, TBT, TBTsum);




IDF = ones(1,size(TBT,2));
for ii = 1:size(TBT,2)
    IDF (ii) = log(trainimgnum /(sum (boolean(TrainBT(:,ii)))+eps)); % this is Wi = log ( N / Ni ) equation in paper
end

save IDF IDF


for i = 1:testimgnum
    
     inlierM = [];
     test_h = TBT(i,:);
     sallimgperword = 0;
     %sum(test_h)
     if b_selkdtree == 1
         
        [vind, d] = vl_kdtreequery (kdforest,TrainBT', double(TBT(i,:)'), 'numneighbors', 1, 'MaxComparisons', 2000);
       
     else


        nonzeroindex = find (test_h~=0  & IDF~=0); 
         Q=repmat(test_h(nonzeroindex).* IDF(nonzeroindex) ,size(TrainBT,1),1);
         D=TrainBT(:,nonzeroindex).*repmat(IDF(nonzeroindex) ,size(TrainBT,1),1);

         sallimgperword = sum( abs(Q) + abs(D) - abs(Q-D) , 2);
        [dis, vind] = sort (sallimgperword,'descend');

     end
     
     gv_listlength=5;
     if b_siftmatch && ~all(trainindex(vind(1))==trainindex(vind(1:gv_listlength)))
         enquiry = testimgname{i};
            if ~b_ransac
                    fr = fopen(strcat(enquiry,filetype), 'r');
                    DESCR1=fread(fr,'double');
                    DESCR1=reshape(DESCR1,length(DESCR1)/feature_dim,feature_dim);
                    DESCR1=DESCR1';
                    fclose(fr);
            end
         %tic
        for pp = 1:gv_listlength
%                 i
%                 pp
             tobecompare = trainimgname{vind(pp)};
             if b_ransac
                 [ipts retnum ratio] = testfund_vgg_my2(enquiry,tobecompare, 0.6);
                 inlierM = [inlierM, ratio];
             else
                       fr = fopen(strcat(tobecompare,filetype), 'r');
                        DESCR2=fread(fr,'double');
                        DESCR2=reshape(DESCR2,length(DESCR2)/feature_dim,feature_dim);
                        DESCR2=DESCR2';
                        fclose(fr);
                        
                 [matches,scores] = vl_ubcmatch(uint8(DESCR1), uint8(DESCR2)); 

                 inlierM = [inlierM, size(matches,2)];

             end
             %pp
        end

         [vv,Mindex] = sort(inlierM,'descend');
         originalindex = vind(Mindex);
     else
         originalindex = vind;
     end
     
     Categind = trainindex(originalindex(1:topN));
     
     Hs = hist (Categind, [1:50]);
     [v, ca] = max (Hs);
     i,ca
     if ca == timgindex (i)
         count = count + 1;
         subcount = subcount + 1;
     end 
     if  i < testimgnum
         if timgindex (i) ~= timgindex (i+1)
         recordcateg = recordcateg + 1;
         subrate = [subrate, subcount/(timgnump(recordcateg)-timgnump(recordcateg-1))]
         subcount = 0;
         end
     else 
         subrate = [subrate, subcount/(timgnump(end)-timgnump(end-1))];
     end
   
end
subrate
classavg=mean(subrate)
rate = (count)/(testimgnum)