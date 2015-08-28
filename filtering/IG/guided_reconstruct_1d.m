function r = guided_reconstruct_1d(g, gx, wx, sigma_s, epsi, sf)

ss = sigma_s;

% reconstruct p from gx
p = g;
cgx = cumsum(gx, 2);
p(:, 2:end) = bsxfun(@plus, p(:, 1), cgx(:, 1:end-1));

q = g;

% 1d guided filter
mean_p = gaussian1d(p, ss);
mean_pq = gaussian1d(p.*q, ss);
mean_q = gaussian1d(q, ss);
cov_pq = mean_pq - mean_p.*mean_q;
mean_p2 = gaussian1d(p.*p, ss);
var_p = mean_p2 - mean_p.^2;
% mean_w = gaussian1d(wx, ss);

a = (cov_pq + epsi.*sf) ./ (var_p + epsi*wx);
b = mean_q - a.*mean_p;

mean_a = gaussian1d(a, ss);
mean_b = gaussian1d(b, ss);

r = mean_a.*p + mean_b;

end
