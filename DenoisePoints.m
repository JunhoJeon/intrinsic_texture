function [ new_Points, new_nMap ] = DenoisePoints(Points, nMap)
%addpath 'E:\2013_2 Intrinsic image decomposition\intrinsic_texture';
[h w c] = size(Points);
% DX = Points(1:h-1, 2:w, :) - Points(2:h, 2:w, :);
% DY = Points(2:h, 1:w-1, :) - Points(2:h, 2:w, :);
% nMap = zeros(h, w, 3);
% nMap(1:h-1, 1:w-1, :) = cross(DX, DY);
% nMap = nMap ./ repmat(sqrt(sum(nMap .* nMap, 3)), [1 1 3]);
% nMap(find(isnan(nMap))) = 0;

prev_Points = Points;
prev_nMap = nMap;

for iter = 1:5
    sigma_c = 0.05;
    sigma_s = 0.01;
    V = prev_Points(1:h, 1:w, :);
    N = prev_nMap(1:h, 1:w, :);
    wSize = 2;
    nlist = [-1 0; 1 0; 0 -1; 0 1; -1 -1; 1 1; -1 1; 1 -1
             -2 0; 2 0; 0 -2; 0 2; -2 -2; 2 2; -2 2; 2 -2;
             -2 -1; -2 1; 2 -1; 2 1; 1 -2; 1 2; -1 2; -1 -2];
    sumMap = zeros(h, w);
    Normalizer = zeros(h, w);
    
    padd_Points = padarray(prev_Points, [2 2], 'symmetric', 'both');
    for i=1:24
        dx = nlist(i, 1); dy = nlist(i, 2);
        Q = padd_Points(1+wSize+dx:h+dx+wSize, 1+wSize+dy:w+dy+wSize, :);
        T = (sum((V-Q) .* (V-Q), 3));
        H = sum(N .* (V-Q), 3);
        Wc = exp(-T ./ (2 * sigma_c * sigma_c));
        Ws = exp(-H.*H ./ (2 * sigma_s * sigma_s));
        sumMap = sumMap + Wc .* Ws .* H;
        Normalizer = Normalizer + Wc .* Ws;
    end
    nD = N .* repmat(sumMap ./ Normalizer, [1 1 3]);
    new_V = V - nD;

    DX = new_V(1:h-1, 2:w, :) - new_V(2:h, 2:w, :);
    DY = new_V(2:h, 1:w-1, :) - new_V(2:h, 2:w, :);
    new_nMap = zeros(h, w, 3);
    new_nMap(1:h-1, 1:w-1, :) = cross(DX, DY);
    new_nMap = new_nMap ./ repmat(sqrt(sum(new_nMap .* new_nMap, 3)), [1 1 3]);
    nanidx = find(isnan(new_nMap));
    new_nMap(nanidx) = prev_nMap(nanidx);
    new_V(nanidx) = prev_Points(nanidx);
    
    prev_Points = new_V;
    prev_nMap = new_nMap;
end
new_Points = new_V;

%new_nMap = tsmooth(new_nMap, 0.005, 20, 0.01);
new_nMap(:,:,1) = medfilt2(new_nMap(:,:,1), [5 5]);
new_nMap(:,:,2) = medfilt2(new_nMap(:,:,2), [5 5]);
new_nMap(:,:,3) = medfilt2(new_nMap(:,:,3), [5 5]);
%TVsmoothed_Normal = tsmooth(nMap, 0.005, 20, 0.01);
%figure;imshow((TVsmoothed_Normal+1)/2);