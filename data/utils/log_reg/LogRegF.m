function fx = LogRegF(data,x)
y = data.y;
A = data.X;
n = size(y,1);

Ax    = (A*x) .* y;
expAx = exp(-Ax);

% evaluate the function values.
fx = sum(log(1+expAx))/n;

end
