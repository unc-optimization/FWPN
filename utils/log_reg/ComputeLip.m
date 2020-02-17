function Lip = ComputeLip(A,AT,muf)

n = size(A,1);
l2nrm = DecoptNormAtAeval('numeric', A, 40, 1e-5, AT);
Lip   = l2nrm/4/n + muf;
end