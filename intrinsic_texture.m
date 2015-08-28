%for sample = [1, 8, 44, 46, 76, 129, 171, 183, 283, 346, 446, 518, 547, 666, 730, 798, 904, 1017, 1039, 1085, 1216, 1272, 1305]
addpath 'regcovsmooth';
addpath 'SimpleModel';
addpath 'RGF';

%%
TestFiles = [171];%, 171, 409, 412, 413, 450, 631, 730, 774, 1137, 1181, 1272, 1305];
tic;
for sample = TestFiles
    tic;
    Scale = 1.0;
    % Read Data & Preprocessing (Structure-texture separation)
    iname = sprintf('image%04d.png', sample);
    dname = sprintf('raw_depth%04d.png', sample);
    %dname = sprintf('depth%04d.png', sample);
    I = im2double(imread(iname));
    d = double(imread(dname))/1000.0;
    D = d;
    [h, w, ~] = size(I);
    
    % Structure-Texture Separation (L.Karacan,E. Erdem and A. Erdem.
    % Structure Preserving Image Smoothing via Region Covariances. ACM
    % Transactions on Graphics (Proceedings of SIGGRAPH Asia 2013), 32(6),
    % November 2013)   
    % you can change this to other texture removing methods (BTF, RGF, etc.)
    %S = regcovsmooth(I,7,4,0.1,'M1');
    S = RollingGuidanceFilter(I, 3, 0.1, 4);
    %%
    D = smooth_d(S, double(D), ones(h, w)); % RGB-D Joint bilateral filtering (Code from Qifen Chen and Vladlen Koltun, ICCV 2013)
    Points=getVectors(size(D,1),size(D,2));
	Points=Points .* D(:,:,[1 1 1]);
    
    % Normal Map Estimation
    [nx, ny, nz]=surfnorm(Points(:,:,1),Points(:,:,2),Points(:,:,3));
    nMap=cat(3,nx,ny,nz);
    Points = Points(7:h-6, 9:w-8, :);
    [Points, nMap] = DenoisePoints(Points, nMap); % Bilateral Mesh Denoising
    %nMap = (im2double(imread('result_normal.png')) * 2-1);
    %nMap = nMap(7:h-6, 9:w-8, :);
    % Cropping (Kinect RGB image has a white padding)
    S = S(7:h-6, 9:w-8, :);
    I = I(7:h-6, 9:w-8, :);

    [h, w, ~] = size(I);
    N = h*w;
%%
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
    i = reshape(sqrt(sum(S.*S, 3)), [N 1]);
    thres = 0.001;
    nthres = 0.001;    
    wr = ones(h, w);
    ws = 1.0;
    sig_c = 0.0001; sig_i = 0.8;
    sig_n = 0.5;
    [consVec, thresMap] = getConstraintsMatrix(C, wr, ws, thres, 0, S, nthres, S);
    % Compute propagation weights (matting Laplacian, surface normal)
    L_S = getLaplacian1(S, zeros(h, w), 0.1^5);    
    nweightMap = getNormalConstraintMatrix(nMap, sig_n);
    % Compute local reflectance constraint (continuous similarity weight)
    [consVecCont, weightMap] = getContinuousConstraintMatrix(C, sig_c, sig_i, S);
%%
    % Optimization
    spI = speye(N, N);    
    mk = zeros(N, 1);
    mk(nNeighbors(:, 1)) = 1;
    mask = spdiags(mk, 0, N, N); % Subsampling mask for local LLE
    mk = zeros(N, 1);
    mk(nNeighbors2(:, 1)) = 1;
    mask2 = spdiags(mk, 0, N, N); % Subsampling mask for non-local LLE
    
    A = 10 * WRC + 2.5 * mask * (spI - LLEGRID) + 2.5 * mask2 * (spI - LLENORMAL) + 10 * L_S + 0.1 * WSC;
    b = 10 * consVecCont;
    disp('Optimizing the system..');
    newS = pcg(A, b, 1e-3, 10000, [], []);
    
    % Visualization and Saving Results
    Shading = reshape(exp(newS), [h w])/2;
    figure;imshow(Shading);
    Reflectance = I ./ repmat(Shading, [1 1 3]) /2;
    figure;imshow(Reflectance);
    toc;

    iname = sprintf('results\\Input%04d.png', sample);
    imwrite(I, iname);
    iname = sprintf('results\\Shading%04d.png', sample);
    imwrite(Shading, iname);
    iname = sprintf('results\\Reflectance%04d.png', sample);
    imwrite(Reflectance, iname);
    iname = sprintf('results\\Result%04d.mat', sample);
    res.I = I;
    res.S = S;
    res.nMap = nMap;
    res.Shading = Shading;
    res.Reflectance = Reflectance;
    save(iname, '-struct', 'res');
end