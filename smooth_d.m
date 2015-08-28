%smooth the depth map to eliminate outliers
function x=smooth_d(im,tmp,mask)
	[m n d]=size(im);
	depth=reshape(tmp,[],1);
	feature=reshape(im,[],d)';
	c=20;
	lambda=0.02;
	row=[1:m*(n-1);m+1:m*n];
	[a b]=ndgrid(1:m-1,1:n);
	tmp=sub2ind([m n],reshape(a,1,[]),reshape(b,1,[]));
	row=[row [tmp;tmp+1]];
	value=exp(-c*sum(abs(feature(:,row(1,:))-feature(:,row(2,:)))))+1e-6;
	if(exist('mask','var')==1)
		mask=reshape(1-double(mask),[],1);
		value(find(mask(row(1,:))|mask(row(2,:))))=1e-3;
	end
	A=sparse(row(1,:),row(2,:),value,m*n,m*n);
	A=A+A';
	D=spdiags(sum(A,2),0,m*n,m*n);
	L=D-A;
	M=spdiags(double(depth>=0.00001),0,m*n,m*n);
	x=(L+lambda*M)\(lambda*M*depth);
	x=reshape(x,m,n);
end
