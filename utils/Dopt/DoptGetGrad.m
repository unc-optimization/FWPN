function grad_val = DoptGetGrad(x, A, At)

[n,m] = size(A);
Z = A*sparse(1:m,1:m,x,m,m)*At;

R = chol(Z);
Rinv = inv(R);
InvZ = Rinv*Rinv';
InvZ = (InvZ+InvZ')/2;
AtInvZA  = At*InvZ*A;
grad_val = -diag(AtInvZA);

end
