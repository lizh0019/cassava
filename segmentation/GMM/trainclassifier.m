function trainclassifier(root,dataset)
%% train classifiers
initialization
load Hist_train

Pathname=strcat(root,dataset,'\');%'D:\databases\101_ObjectCategories\'
Category=dir(strcat(Pathname,'*.*'));%read the database

%obtain offset
% offset=[0];
% 
% for Ii=1:CategoryNum,
%     categoryimgnum=0;
%     if (Category(Ii).isdir==1 && ~strcmp(Category(Ii).name,'.') && ~strcmp(Category(Ii).name,'..')), 
%         foldername=Category(Ii).name;
%         Image=dir(strcat(Pathname,foldername,'\*.jpg'));
%         categoryimgnum=categoryimgnum+length(Image);
%         offset=[offset,offset(end)+categoryimgnum];
%     end
% end
load offset
% nin = size(Hist,2);
% kernel = 'X2';%rbf X2 rbffull linear poly
% kernelpar = 0.1;
% net = svm(nin,kernel,kernelpar,100);
% net=repmat(net,CategoryNum-2,1);
% for i=1:CategoryNum-2,net(i).qpsize=size(Hist,1);end
% 
% %train different classifiers
% for i=1:CategoryNum-2,
%     Y = -1*ones(offset(end),1);
%     pos=offset(i+1);
%     Y(offset(i)+1:pos) = 1; 
%     net(i).FCoeff = [ones(offset(i),1);1*ones(offset(i+1)-offset(i),1);ones(totalimgnum-offset(i+1),1)]; % for FSVM
%     net(i) = svmtrain(net(i), Hist, Y, [], 0);
% end
% save NET net



yapp = zeros(offset(end),1);
for i=1:CategoryNum-2,
    pos=offset(i+1);
    yapp(offset(i)+1:pos) = i; 
end
%-------------------------------------------------------
% Kernel Parameters
%-------------------------------------------------------
%kernel='htrbf';
%kernel='gaussian';
%kernel='x2';
kernel='intersection';
% kernel 	: kernel function
%		Type								Function					Option
%		Polynomial						'poly'					Degree (<x,xsup>+1)^d
%		Homogeneous polynomial		'polyhomog'				Degree <x,xsup>^d	
%		Gaussian							'gaussian'				Bandwidth
%		Heavy Tailed RBF				'htrbf'					[a,b]   %see Chappelle 1999	
%		Mexican 1D Wavelet 			'wavelet'
%		Frame kernel					'frame'
%		'sin','numerical'...	
% case 'gaussian'
%     [nk,nk2]=size(kerneloption);
%     if nk ~=nk2
%         if nk>nk2
%             kerneloption=kerneloption';
%         end;
%     else
%         kerneloption=ones(1,n2)*kerneloption;
%     end;
%     
%     if length(kerneloption)~=n2 & length(kerneloption)~=n2+1 
%         error('Number of kerneloption is not compatible with data...');
%     end;
%     
%     
%     metric = diag(1./kerneloption.^2);
%     ps = x*metric*xsup'; 
%     [nps,pps]=size(ps);
%     normx = sum(x.^2*metric,2);
%     normxsup = sum(xsup.^2*metric,2);
%     ps = -2*ps + repmat(normx,1,pps) + repmat(normxsup',nps,1) ; 
%     
%     
%     K = exp(-ps/2);
%     
% case 'htrbf'    % heavy tailed RBF  %see Chappelle Paper%
%     b=kerneloption(2);
%     a=kerneloption(1);
%     for i=1:n
%         ps(:,i) = sum( abs((x.^a - ones(n1,1)*xsup(i,:).^a)).^b   ,2);
%     end;
%     
%     
%     K = exp(-ps);
%     
% case 'x2'
% %        dist2=0.5*(X1-X2).^2/(X1+X2+eps);
%     for i=1:n
%         ps(:,i) = sum( (x- ones(n1,1)*xsup(i,:)).^2 / (x+ repmat(xsup(i,:),n1,1)+eps)   ,2);
%     end;
%     ps=ps/max(max(ps));
%     
%     K = exp(-120*ps);
% 
% case 'gaussianslow'    %
%     %b=kerneloption(2);
%     %a=kerneloption(1);
%     for i=1:n
%         ps(:,i) = sum( abs((x - ones(n1,1)*xsup(i,:))).^2 ,2)./kerneloption.^2/2;
%     end;
%     
%     
%     K = exp(-ps);
% case 'multiquadric'
%     metric = diag(1./kerneloption);
%     ps = x*metric*xsup'; 
%     [nps,pps]=size(ps);
%     normx = sum(x.^2*metric,2);
%     normxsup = sum(xsup.^2*metric,2);
%     ps = -2*ps + repmat(normx,1,pps) + repmat(normxsup',nps,1) ; 
%     K=sqrt(ps + 0.1);
% case 'wavelet'
%     K=kernelwavelet(x,kerneloption,xsup);     
% case 'frame'
%     K=kernelframe(x,kerneloption,xsup,framematrix,vector,dual);
% case 'wavelet2d'
%     K=wav2dkernelint(x,xsup,kerneloption);
% case 'radialwavelet2d'
%     K=radialwavkernel(x,xsup);    
% case 'tensorwavkernel'
%     [K,option]=tensorwavkernel(x,xsup,kerneloption);  
% 
% case 'numerical'
%     K=kerneloption.matrix;
% case 'polymetric'
%     K=x*kerneloption.metric*xsup';
%     
% case 'jcb'
%     K=x*xsup';
    
    
%kerneloption=[1];
kerneloption=[10,0.1];
C=100000;
verbose=1;
lambda=1e-7;
nbclass=CategoryNum-2;
%-------------------------------------------------------
% Solving
%-------------------------------------------------------
%yapp(end)=3;nbclass=3;
net.CategoryNum=CategoryNum-2;
%[xsup,w,b,nbsv,pos,alpha]=svmmulticlass(Hist,yapp,nbclass,C,lambda,kernel,kerneloption,verbose);%at least 3 classes
cd 'D:\program_library\SVM\'
[xsup,w,b,nbsv]=svmmulticlassoneagainstall(Hist,yapp,nbclass,C,lambda,kernel,kerneloption,verbose);
net.xsup=xsup;net.w=w;net.b=b;net.nbsv=nbsv;net.kernel=kernel;net.kerneloption=kerneloption;
%[ypred] = svmmultival(Hist,xsup,w,b,nbsv,kernel,kerneloption);
[ypred,maxi] = svmmultival(Hist,net.xsup,net.w,net.b,net.nbsv,net.kernel,net.kerneloption);
fprintf( '\nRate of correct class in training data : %2.2f \n',100*sum(ypred==yapp)/length(yapp)); 
cd 'D:\content_context\BoWBaseline_entropy'

save NET net

