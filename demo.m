addpath('filtering/RGF');                     % rolling guidance filter
addpath('filtering/regcovsmooth');            % region covariance filter

% Read Data & Preprocessing (Structure-texture separation)
I = im2double(imread('images/image0171.png'));
depth = double(imread('images/raw_depth0171.png'))/1000.0;
%depth(:) = 0;
% I = im2double(imread('image0001.png'));
% load('Z0001.mat');
% depth = Z;

% Structure-Texture Separation (L.Karacan,E. Erdem and A. Erdem.
% Structure Preserving Image Smoothing via Region Covariances. ACM
% Transactions on Graphics (Proceedings of SIGGRAPH Asia 2013), 32(6),
% November 2013)   
% you can change this to other texture removing methods (BTF, RGF, etc.)
tic;
% S = regcovsmooth(I,7,4,0.1,'M1');
S = RollingGuidanceFilter(I, 3, 0.1, 4);
%%
[reflectance, shading] = intrinsic_decomp(I, S, depth, 0.0001, 0.8, 0.5);
toc;
figure;imshow(reflectance);
figure;imshow(shading);