function S = regcovsmooth(Im,radius,ps,sigma,model)
%   Paper : "Structure-Preserving Image Smoothing via Region Covariances"
%   Author: Levent Karacan, Erkut Erdem, Aykut Erdem 
%            (karacan@cs.hacettepe.edu.tr, erkut@cs.hacettepe.edu.tr, aykut@cs.hacettepe.edu.tr)
%   Date  : 11/05/2013
%   Version : 1.0 
%   Copyright 2013, Hacettepe University, Turkey.
%
%   Output:
%   S          : Structure component extracted from input image I.
%
%   Parameters:
%   @Im        : Input image.
%   @radius    : Filter size=2*radius X 2*radius.
%   @ps        : Patch size=2*ps X 2*ps
%   @sigma     : Parameter specifying smoothness amount.
%                Range (0.05, 1], 0.1 by defalut.
%   @model     : Model selection 'M1','M2','M3'
%
%   Example
%   ==========
%   I  = imread('zeugma.jpg');
%   S  = regcovsmooth(I);
%   Default Parameters (radius = 10, ps = 4, sigma = 0.2, model='M1')
%   figure, imshow(I), figure, imshow(S);

if (~exist('radius','var'))
    radius=10;
end
if (~exist('ps','var'))
    ps=4;
end
if (~exist('sigma','var'))
    sigma = 0.2;
end
if (~exist('model','var'))
    model='M2';
end

padsize=radius+ps;
filterlength=(2*radius+1)*(2*radius+1);
filtersize=(2*radius+1);
patchsize=2*ps+1;

[~,~,channel]=size(Im);
Im=double(Im);

if channel==3
    Img = mean(double(Im),3);
else
    Img = double(Im);
end

% Extract features
Im=padarray(Im,[padsize padsize],'symmetric');
Img=padarray(Img,[padsize padsize],'symmetric');
[h,w,ch]=size(Img);
I = Img;
d = [-1 0 1];
Iy = imfilter(Im,d,'symmetric','same','conv');
Ix = imfilter(Im,d','symmetric','same','conv');
Iyy = imfilter(Iy,d,'symmetric','same','conv');
Ixx = imfilter(Ix,d','symmetric','same','conv');
Ix=max(abs(Ix),[],3);
Iy=max(abs(Iy),[],3);
Ixx=max(abs(Ixx),[],3);
Iyy=max(abs(Iyy),[],3);
[s2, s1] = meshgrid(1:w,1:h);

I = Img/max(Img(:));
dummy=abs(Ix);
dummy2=abs(Iy);
dummy3=abs(Ixx);
dummy4=abs(Iyy);
Ixx = abs(Ixx)/max(dummy3(:));
Iyy = abs(Iyy)/max(dummy4(:));
Ix = abs(Ix)/max(dummy(:));
Iy = abs(Iy)/max(dummy2(:));
s1 = s1/max(s1(:));
s2 = s2/max(s2(:));
sF=7; % feature size
sFF=sF*sF;

% Model 1
if (strcmp(model,'M1'))
    regcovNew=zeros(sF,sF,h-2*ps,w-2*ps);
    regmeanNew=zeros(1,sF,h-2*ps,w-2*ps);
    Chols=zeros(sF,sF,h-2*ps,w-2*ps);
    birim=eye(sF,sF)*0.001;
    alpha=sqrt(8.0);
    
    for i=1+ps:1:h-ps
        for j=1+ps:1:w-ps
            patch=I(i-ps:i+ps,j-ps:j+ps);
            Ixp=Ix(i-ps:i+ps,j-ps:j+ps);
            Iyp=Iy(i-ps:i+ps,j-ps:j+ps);
            Ixxp=Ixx(i-ps:i+ps,j-ps:j+ps);
            Iyyp=Iyy(i-ps:i+ps,j-ps:j+ps);
            sp1=s1(i-ps:i+ps,j-ps:j+ps);
            sp2=s2(i-ps:i+ps,j-ps:j+ps);
            mat(:,1)=patch(:);
            mat(:,2)=abs(Ixp(:));
            mat(:,3)=abs(Iyp(:));
            mat(:,4)=abs(Ixxp(:));
            mat(:,5)=abs(Iyyp(:));
            mat(:,6)=sp2(:);
            mat(:,7)=sp1(:);
            
            covM=cov(mat);
            mmean=mean(mat);
            
            regcovNew(:,:,i-ps,j-ps)=covM;
            regmeanNew(:,:,i-ps,j-ps)=mmean;
            
            Cov1 = covM +birim;
            Cov1 = alpha*(size(Cov1,1)+0.1)*Cov1;
            Chols(:,:,i-ps,j-ps)=chol(Cov1);
        end
    end
    filterim=zeros(h-2*padsize,w-2*padsize,ch);
    
    for i=1+padsize:1:h-padsize
        for j=1+padsize:1:w-padsize
            covC=regcovNew(:,:,i-ps,j-ps);
            m=regmeanNew(:,:,i-ps,j-ps);
            
            covC = covC + 0.001*eye(size(covC));
            covC = alpha*(size(covC,1)+0.1)*covC;
            L = chol(covC);
            li=bsxfun(@plus,L',m);
            lj=bsxfun(@minus,m,L');
            li=li';
            lj=lj';
            li=li(:);
            lj=lj(:);
            
            resRef = [m li' lj'];
            
            REGMEAN2=regmeanNew(:,:,i-radius-ps:i+radius-ps,j-radius-ps:j+radius-ps);
            REGMEAN2=reshape(REGMEAN2,[sF filterlength]);
            REG=kron(REGMEAN2,ones(1,sF));
            REG=reshape(REG,[sF sF filterlength]);
            
            LM=Chols(:,:,i-radius-ps:i+radius-ps,j-radius-ps:j+radius-ps);
            LM=LM(:,:,:);
            LI=LM+REG;
            LJ=REG-LM;
            LI=reshape(LI,[sFF filterlength])';
            LJ=reshape(LJ,[sFF filterlength])';
            REF=[REGMEAN2' LI LJ];
            
            d1=bsxfun(@minus,REF,resRef);
            d2=d1.^2;
            DISTMAT=sum(d2,2);
            
            weightM=exp(-((DISTMAT.^2.0)./(2.0*sigma^2)));
            weightM=weightM./sum(weightM(:));
            
            if channel==3
                region1=Im(i-radius:i+radius,j-radius:j+radius,1);
                region2=Im(i-radius:i+radius,j-radius:j+radius,2);
                region3=Im(i-radius:i+radius,j-radius:j+radius,3);
                filterim(i-padsize,j-padsize,1) = sum(region1(:).*weightM);
                filterim(i-padsize,j-padsize,2) = sum(region2(:).*weightM);
                filterim(i-padsize,j-padsize,3) = sum(region3(:).*weightM);
            else
                region1=Im(i-radius:i+radius,j-radius:j+radius);
                filterim(i-padsize,j-padsize,1) = sum(region1(:).*weightM);
            end
        end
    end
    S=filterim;

% Model 2
elseif strcmp(model,'M2')
    filterim=zeros(h-2*padsize,w-2*padsize,ch);
    regcovNew=zeros(sF,sF,h-2*ps,w-2*ps);
    regmeanNew=zeros(1,sF,h-2*ps,w-2*ps);
 
    % Compute region covariances
    for i=1+ps:1:h-ps
        for j=1+ps:1:w-ps
            patch=I(i-ps:i+ps,j-ps:j+ps);
            Ixp=Ix(i-ps:i+ps,j-ps:j+ps);
            Iyp=Iy(i-ps:i+ps,j-ps:j+ps);
            Ixxp=Ixx(i-ps:i+ps,j-ps:j+ps);
            Iyyp=Iyy(i-ps:i+ps,j-ps:j+ps);
            [sp2,sp1]=meshgrid(1:size(patch,2),1:size(patch,1));
            
            mat(:,1)=patch(:);
            mat(:,2)=abs(Ixp(:));
            mat(:,3)=abs(Iyp(:));
            mat(:,4)=abs(Ixxp(:));
            mat(:,5)=abs(Iyyp(:));
            mat(:,6)=sp2(:);
            mat(:,7)=sp1(:);
            
            covM=cov(mat);
            mmean=mean(mat);
            
            regcovNew(:,:,i-ps,j-ps)=covM;
            regmeanNew(:,:,i-ps,j-ps)=mmean;            
        end
    end
    
  
    clear Ix Iy Ixx Iyy I mat
    UNITMAT= reshape(repmat(eye(sF),1,filtersize*filtersize),[sF sF filtersize*filtersize]);
    for i=1+padsize:1:h-padsize
        for j=1+padsize:1:w-padsize
            cov1=regcovNew(:,:,i-ps,j-ps);
            m1=regmeanNew(:,:,i-ps,j-ps);
            
            REGCOV2=regcovNew(:,:,i-radius-ps:i+radius-ps,j-radius-ps:j+radius-ps);
            REGCOV2=REGCOV2(:,:,:);
            
            REGMEAN2=regmeanNew(:,:,i-radius-ps:i+radius-ps,j-radius-ps:j+radius-ps);
            REGMEAN2=reshape(REGMEAN2,[sF filtersize*filtersize]);
            REGMEAN2=REGMEAN2';
            
            cov1=cov1*0.5;
            REGCOV=bsxfun(@plus,REGCOV2*0.5,cov1);
            MEANDIFF=bsxfun(@minus,REGMEAN2,m1);
            MEANDIFFTRANS=MEANDIFF';
            
            REGCOV=mmx('backslash',REGCOV,UNITMAT); % uses LU
            
            D1=reshape(MEANDIFF',[1 sF filtersize*filtersize]);
            D2=reshape(MEANDIFFTRANS,[sF 1 filtersize*filtersize]);
            dist1=mtimesx(D1,REGCOV);
            DISTMAT=mtimesx(dist1,D2);
            DISTMAT=DISTMAT(:);
            
            % Weights
            weight=exp(-((DISTMAT.^2)./(2*sigma^2)));
            weight=weight./sum(weight(:));
            
            if channel==3
                region1=Im(i-radius:i+radius,j-radius:j+radius,1);
                region2=Im(i-radius:i+radius,j-radius:j+radius,2);
                region3=Im(i-radius:i+radius,j-radius:j+radius,3);
                filterim(i-padsize,j-padsize,1) = sum(region1(:).*weight);
                filterim(i-padsize,j-padsize,2) = sum(region2(:).*weight);
                filterim(i-padsize,j-padsize,3) = sum(region3(:).*weight);
            else
                region1=Im(i-radius:i+radius,j-radius:j+radius);
                filterim(i-padsize,j-padsize,1) = sum(region1(:).*weight);
            end
        end
    end
    S=filterim;

% Model 3 (Using KL-div btw 2 normal distributions, suggested by Rahul Narain)
elseif (strcmp(model,'M3'))
    sigma=2*sigma;
    filterim=zeros(h-2*padsize,w-2*padsize,ch);
    regcov=zeros(sF,sF,h-2*ps,w-2*ps);
    regmean=zeros(1,sF,h-2*ps,w-2*ps);
    regcovInv=zeros(sF,sF,h-2*ps,w-2*ps);
    regcovDet=zeros(h-2*ps,w-2*ps);
    
    for i=1+ps:1:h-ps
        for j=1+ps:1:w-ps
            patch=I(i-ps:i+ps,j-ps:j+ps);
            Ixp=Ix(i-ps:i+ps,j-ps:j+ps);
            Iyp=Iy(i-ps:i+ps,j-ps:j+ps);
            Ixxp=Ixx(i-ps:i+ps,j-ps:j+ps);
            Iyyp=Iyy(i-ps:i+ps,j-ps:j+ps);
            [sp2,sp1]=meshgrid(1:size(patch,2),1:size(patch,1));
            
            mat(:,1)=patch(:);
            mat(:,2)=abs(Ixp(:));
            mat(:,3)=abs(Iyp(:));
            mat(:,4)=abs(Ixxp(:));
            mat(:,5)=abs(Iyyp(:));
            mat(:,6)=sp2(:);
            mat(:,7)=sp1(:);
            
            covM=cov(mat);
            mmean=mean(mat);
            regcov(:,:,i-ps,j-ps)=covM;
            regmean(:,:,i-ps,j-ps)=mmean;
            regcovInv(:,:,i-ps,j-ps)=covM\eye(sF,sF);
            regcovDet(i-ps,j-ps)=det(covM);
        end
    end
    
    
    unitmat=eye(sF,sF);
    for i=1+padsize:1:h-padsize
        for j=1+padsize:1:w-padsize
            cov1=regcov(:,:,i-ps,j-ps);
            m1=regmean(:,:,i-ps,j-ps);
            Det1=regcovDet(i-ps,j-ps);
            
            InvCOV2=regcovInv(:,:,i-ps-radius:i-ps+radius,j-ps-radius:j-ps+radius);
            InvCOV2=InvCOV2(:,:,:);
            
            COV1=repmat(cov1,1,filterlength);
            COV1=reshape(COV1,[sF sF filterlength]);
            T=mtimesx(InvCOV2,COV1);
            
            T=bsxfun(@times,T,unitmat);
            T1=sum(sum(T,1),2);
            T1=T1(:);
            
            REGMEAN2=regmean(:,:,i-radius-ps:i+radius-ps,j-radius-ps:j+radius-ps);
            MEANDIFFMat=bsxfun(@minus,REGMEAN2,m1);
            MEANDIFF=reshape(MEANDIFFMat,[1 sF ,filterlength]);
            MEANDIFFTRANS=reshape(MEANDIFFMat,[sF 1 ,filterlength]);
            
            dist1=mtimesx(MEANDIFF,InvCOV2);
            T2=mtimesx(dist1,MEANDIFFTRANS);
            T2=T2(:);
            T3=ones(filterlength,1)*sF;
            DET2=regcovDet(i-radius-ps:i+radius-ps,j-radius-ps:j+radius-ps);
            T4=log(Det1./DET2);
            T4=T4(:);
            DISTMAT=(T1+T2-T3-T4)/2;
            
            weight=exp(-((DISTMAT)./(2*sigma^2)));
            weight=weight./sum(weight(:));
            
            if channel==3
                region1=Im(i-radius:i+radius,j-radius:j+radius,1);
                region2=Im(i-radius:i+radius,j-radius:j+radius,2);
                region3=Im(i-radius:i+radius,j-radius:j+radius,3);
                filterim(i-padsize,j-padsize,1) = sum(region1(:).*weight);
                filterim(i-padsize,j-padsize,2) = sum(region2(:).*weight);
                filterim(i-padsize,j-padsize,3) = sum(region3(:).*weight);
            else
                region1=Im(i-radius:i+radius,j-radius:j+radius);
                filterim(i-padsize,j-padsize,1) = sum(region1(:).*weight);
            end
        end
    end
    S=filterim;
end