function grad_val = LogRegGetGrad(x, A, C, muf, n)
Ax = A*x;
temp = 1 + exp(Ax);
inv_temp = 1./temp;
temp1 = 1/n*(1 - inv_temp);
grad_val = (temp1'*A)' + muf*C'*(C*x);
end


