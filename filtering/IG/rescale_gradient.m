function [r_gx, wx] = rescale_gradient(I, ss, do_normalize)

if ~exist('do_normalize', 'var'), do_normalize = false; end

% ss = ss*sqrt(2);
fr = ceil(3*ss);

% gaussian patch graidient
kl = -fr+1:0;
kl = exp(-0.5*(kl/ss).^2);
kl = kl / sum(kl);
k = [0 -kl kl(end:-1:1)];

% exponential function
% tval = 0.05;
% fr = ceil(3*ss);
% alpha = tval^(1/(3*ss));
% kl = -fr+1:0;
% kl = alpha.^abs(kl);
% kl=  kl / sum(kl);
% k = [0 -kl kl(end:-1:1)];

% derivative of gaussian
% x = -fr:fr;
% k = exp(-0.5*(x/ss).^2);
% k = x .* k / (ss.^2);
% k = k / sum(abs(k)) * 2;
% k(2:fr+1) = k(1:fr);
% k(1) = 0;

ky = fspecial('gaussian', [2*fr+1 1], ss);

% k = padarray(k, [0 fr]);
% k = imfilter(k, ky.', 'symmetric');
% k = k / (sum(abs(k)));
% k = k * 2;

gx = imfilter(I, [0 -1 1], 'replicate');
px = imfilter(I, k, 'replicate');

% RTV test
% lx = abs(imfilter(gx, ky.', 'symmetric'));
% dx = imfilter(abs(gx), ky.', 'symmetric');
% mean_lx = mean(lx, 3);
% mean_dx = mean(dx, 3);
% 
% w = repmat(min(1, (abs(mean_lx)+eps) ./ (abs(mean_dx)+eps)), [1 1 size(I, 3)]);
% r_gx = gx .* w;


% px = imfilter(px, ky, 'symmetric');
% gx = imfilter(gx, ky.', 'symmetric');
% px = imfilter(px, ky.', 'symmetric');

% mean_px = mean(px, 3);
% mean_gx = mean(gx, 3);

nch = size(I, 3);

[~, midx] = max(abs(px), [], 3);

[hh, ww, ~] = size(px);
pidx = reshape(1:hh*ww, [hh ww]);
midx = (midx-1)*hh*ww + pidx;

mean_px = px(midx);
mean_gx = gx(midx);

% mean_px = px(:, :, 1);
% mean_gx = gx(:, :, 1);
% for ch = 2:nch
%     Ich = px(:, :, ch);
%     idx = abs(Ich) > abs(mean_px);
%     mean_px(idx) = Ich(idx);
%     Ich = gx(:, :, ch);
%     idx = abs(Ich) > abs(mean_gx);
%     mean_gx(idx) = Ich(idx);
% end

%w = repmat((abs(mean_px)+0.0001) ./ (abs(mean_gx)+0.0001), [1 1 nch]);
w = repmat((abs(mean_px)+0.001) ./ (abs(mean_gx)+0.04), [1 1 nch]);
% w = repmat(min(1, (abs(mean_px)+eps) ./ (abs(mean_gx)+eps)), [1 1 nch]);
% w = imfilter(w, ky.', 'symmetric');

% gx = repmat(mean_gx, [1 1 nch]);
pidx = repmat(mean_gx.*mean_px < 0, [1 1 nch]);
r_gx = gx .* min(w, 1);

%rescaleIdx = (abs(mean_px)+0.001) > (abs(mean_gx)+0.04);
%r_gx = bsxfun(@times, gx, rescaleIdx) + bsxfun(@times, px, (~rescaleIdx));
r_gx(pidx) = 0;
%gx .* repmat(((), [1 1 3]) + px .* repmat((abs(mean_px) <= abs(mean_gx)), [1 1 3]);

% gx = pgx;

% normalizing each patch makes smoother result 
if do_normalize
  l0 = mean(abs(gaussian1d(gx, ss)), 3);
  l = mean(abs(gaussian1d(r_gx, ss)), 3);
  a = min(1, (l0+eps) ./ (l+eps));
  a = gaussian1d(a, ss);

  r_gx = bsxfun(@times, r_gx, a);
end

wx = gaussian1d(1./w(:, :, 1), ss);
% wx = 1./w(:, :, 1);

end