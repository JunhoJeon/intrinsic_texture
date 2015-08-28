function [S, et] = pgfilter(img, sigma_s, epsi, N)

I = gpuArray(im2single(img));
%  I = im2single(img);

Nd = 3;

S = I;

tic

for ii = 0:N-1
   [ngx, wx] = rescale_gradient(S, sigma_s, false);
   [ngy, wy] = rescale_gradient(permute(S, [2 1 3]), sigma_s, false);
%  [ngx, ngy] = rescale_gradient_xy(S, sigma_s);
%  ngy = permute(ngy, [2 1 3]);
  
  % handle each channel separately
  for ch = 1:size(S, 3)
    S(:, :, ch) = reconstruct_img(S(:, :, ch), ngx(:, :, ch), ngy(:, :, ch), wx, wy, sigma_s, Nd, epsi, 0);
%     S(:, :, ch) = reconstruct_img_2d(S(:, :, ch), ngx(:, :, ch), ngy(:, :, ch), wx, wy, sigma_s, epsi);
  end
  
%   epsi = epsi / 2;
%   if epsi <= 0.01^2 / 2, break; end
%   epsi = max(0.01^2, epsi);
%   sigma_s = max(0.5, sigma_s / 2);
  
  %figure(101), imshow(S);
  %drawnow;

%   sf = sf / 2;
  
end

toc

end