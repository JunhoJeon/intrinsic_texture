function B = gaussian1d(I, ss)

fr = ceil(3*ss);
B = imfilter(I, fspecial('gaussian', [1 2*fr+1], ss), 'symmetric');

end