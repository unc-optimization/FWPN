function obj_val = LogRegGetObj(x, A, C, muf, n)
temp = A*x;
temp1 = C*x;
obj_val = 1/n*sum(log(1+ exp(temp))) + muf*sum(temp1.^2)/2;
end


