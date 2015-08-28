function r_g = reconstruct_img(g, gx, gy, wx, wy, sigma_s, n_iter, epsi, sf)

r_g = g;

% iterate x and y
for ii = 0:n_iter-1
  ss_i = sigma_s * sqrt(3) * 2^(n_iter - (ii + 1)) / sqrt(4^n_iter - 1);
  r_g = guided_reconstruct_1d(r_g, gx, wx, ss_i, epsi, sf);
%   imwrite(gather(r_g), num2str(ii, 'recon_x_%d.jpg'), 'quality', 85);
  r_g = permute(r_g, [2 1 3]);
  r_g = guided_reconstruct_1d(r_g, gy, wy, ss_i, epsi, sf);
  r_g = permute(r_g, [2 1 3]);
%   imwrite(gather(r_g), num2str(ii, 'recon_y_%d.jpg'), 'quality', 85);
end

end

% 
% function r_g = filter_horizontal(g, gx, ss_i)
% 
% fr = ceil(3*ss_i);
% kx = fspecial('gaussian', [1 2*fr+1], ss_i);
% 
% cgx = cumsum(gx, 2);
% 
% temp_g = repmat(g(:, 1), [1 size(g, 2)]);
% temp_g(:, 2:end) = temp_g(:, 2:end) + cgx(:, 1:end-1);
% 
% p_g = padarray(g, [0 fr], 'symmetric');
% p_tg = padarray(temp_g, [0 fr], 'symmetric');
% pl = fr+1;
% pr = fr+size(g, 2);
% 
% r_g = zeros(size(g), 'like', g);
% for x = -fr:fr
%   w_s = kx(fr+x+1);
%   d_g = p_tg(:, pl+x:pr+x) - temp_g;
%   
%   r_g = r_g + w_s * (p_g(:, pl+x:pr+x) - d_g);
% end
% 
% end