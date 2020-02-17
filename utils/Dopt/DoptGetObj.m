function obj_val = DoptGetObj(x, A, At)

[n,m] = size(A);
Z = A*sparse(1:m,1:m,x,m,m)*At;

R = chol(Z);
obj_val = -sum(log(diag(R).^2));

end
