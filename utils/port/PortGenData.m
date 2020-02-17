% [M,m,W]    = fn_port_datagen(1000, 1500, 0.1, 0.5)
function W = PortGenData(n, p, tsig, rat)

if nargin == 3
    rat = 0.5;
end

W                   = ones(n,p);
X                   = rand(n,p);
X(X >= rat)         = -1;
X(X ~=-1)           = 1;
Inc                 = tsig / 2 * randn(n,p);
Inc                 = abs(Inc) .* X;
W                   = W + Inc;
end