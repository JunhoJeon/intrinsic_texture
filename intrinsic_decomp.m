function [ res_r, res_s ] = intrinsic_decomp( I, S, depth, sigma_c, sigma_i, sigma_n)
%   Matlab and mex implementation of "Intrinsic Image Decomposition using
%   Texture Separation and Surface Normal". It require RGB image, its
%   texture-removed image, and aligned depth image as inputs. It decompose
%   input image into reflectance and shading image.
%
%   Parameters
%   I: input RGB image (double)
%   S: texture-removed RGB image (in original paper, we use regcovsmooth.
%   depth: Kinect depth image (in meter).
%   sigma_c, sigma_i, sigma_n: refer the below paper. Basically control the
%   variance of chroma, normal, intensity similarity funtions, but
%   modification is not recommended.
%
%   ==========
%   The Code is created based on the method described in the following paper:
%   [1] "Intrinsic Image Decomposition Using Structure-Texture Separation
%   and Surface Normals", Junho Jeon, Sunghyun Cho, Xin Tong, Seungyong
%   Lee, European Conference on Computer Vision (ECCV), 2014
%
%   The code and the algorithm are for non-comercial use only.
%
%  
%   Author: Junho Jeon (zwitterion27@postech.ac.kr)
%   Date  : 08/28/2015
%   Version : 1.0 

addpath('mex', 'utils');            % basic utilities
    if ~exist('sc','var')
        sigma_c = 0.0001;
    end

    if ~exist('si','var')
        sigma_i = 0.8;
    end

    if ~exist('sn','var')
        sigma_n = 0.5;
    end


    D = depth;
    [h, w, ~] = size(I);
    
    D = smooth_d(S, double(D), ones(h, w)); % RGB-D Joint bilateral filtering (Code from Qifen Chen and Vladlen Koltun, ICCV 2013)
    Points=getVectors(size(D,1),size(D,2));
	Points=Points .* D(:,:,[1 1 1]);
    
    % Normal Map Estimation
    [nx, ny, nz]=surfnorm(Points(:,:,1),Points(:,:,2),Points(:,:,3));
    nMap=cat(3,nx,ny,nz);
    Points = Points(7:h-6, 9:w-8, :);           % cropping the image
    [Points, nMap] = DenoisePoints(Points, nMap); % Bilateral Mesh Denoising
    %nMap = (im2double(imread('result_normal.png')) * 2-1);
    %nMap = nMap(7:h-6, 9:w-8, :);
    % Cropping (Kinect RGB image has a white padding)
    S = S(7:h-6, 9:w-8, :);
    I = I(7:h-6, 9:w-8, :);

    [h, w, ~] = size(I);
    N = h*w;

    % Minimum Patch Normal Variance Sub-Sampling
    var_pad = 2; var_patch = 5;
    varianceOfNormalMap = var(im2col(padarray(nMap(:, :, 1), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
    varianceOfNormalMap = varianceOfNormalMap + var(im2col(padarray(nMap(:, :, 2), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
    varianceOfNormalMap = varianceOfNormalMap + var(im2col(padarray(nMap(:, :, 3), [var_pad var_pad], 'symmetric'), [var_patch var_patch], 'sliding'));
    vMap = reshape(varianceOfNormalMap, [h w]);
    vMap_p = permute(vMap, [2 1 3]);
    nMap_p = nMap;
    % Compute Sub-Sampled non-local LLE Constraint
    nNeighbors2 = getGridLLEMatrixNormal(nMap_p, vMap_p, 50, 3, 12);   
    % Compute Sub-Sampled local LLE Constraint 
    nMap_p(:,:,4:6) = Points;
    nNeighbors = getGridLLEMatrix(nMap_p, vMap_p, 50, 6, 12);
    
    C = getChrom(S);
    thres = 0.001;
    nthres = 0.001;    
    wr = ones(h, w);
    ws = 1.0;
    sig_c = sigma_c; sig_i = sigma_i;
    sig_n = sigma_n;
    [consVec, thresMap] = getConstraintsMatrix(C, wr, ws, thres, 0, S, nthres, S);
    % Compute propagation weights (matting Laplacian, surface normal)
    disp('computing laplacian matrix.');
    L_S = getLaplacian1(S, zeros(h, w), 0.1^5);    
    nweightMap = getNormalConstraintMatrix(nMap, sig_n);
    % Compute local reflectance constraint (continuous similarity weight)
    [consVecCont, weightMap] = getContinuousConstraintMatrix(C, sig_c, sig_i, S);
    
    % Optimization
    spI = speye(N, N);    
    mk = zeros(N, 1);
    mk(nNeighbors(:, 1)) = 1;
    mask = spdiags(mk, 0, N, N); % Subsampling mask for local LLE
    mk = zeros(N, 1);
    mk(nNeighbors2(:, 1)) = 1;
    mask2 = spdiags(mk, 0, N, N); % Subsampling mask for non-local LLE
    A = 4 * WRC + 3 * mask * (spI - LLEGRID) + 3 * mask2 * (spI - LLENORMAL) + 1 * L_S + 0.025 * WSC;
    b = 4 * consVecCont;
    disp('Optimizing the system...');
    newS = pcg(A, b, 1e-3, 10000, [], []);
    % Visualization and Saving Results
    res_s = reshape(exp(newS), [h w])/2;
    res_r = I ./ repmat(res_s, [1 1 3]) /2;
end

