%compute the view vector at each pixel
function p=getVectors(m,n,fov)
	if(~exist('fov','var'))
		fov=60;
	end
	x=((1:n)-(n+1)/2)/(n/2)*tan(fov/2/180*pi);
	y=-((1:m)-(m+1)/2)/(m/2)*tan(fov/2/180*pi)*(m/n);
	p=zeros(m,n,3);
	for i=1:m
		p(i,:,2)=y(i);
	end
	for i=1:n
		p(:,i,1)=x(i);
	end
	p(:,:,3)=-1;
end