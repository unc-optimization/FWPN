function gx = LogisticG(data,x)
y = data.y;
A = data.X;
n = size(y,1);

Ax    = (A*x) .* y;
expAx = exp(-Ax);

% evaluate the gradient.
px  = 1./(1+expAx)-1;
ypx = y.*px;
gx  = A'*ypx/n;
end