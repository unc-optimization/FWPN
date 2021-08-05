function [y, lambda, info] = PortIVMSubSolver(x0, y0, W, theta, max_iter)
%% Conditional Gradient Method to solve the subproblem of portfolio:
%%
%%      min  <grad_f(x0), u> + 1/2< hessian_f(x0)*(u - x0), u - x0>
%%      s.t. u in Simplex.
%%
%% grad_f(x)    := -WT(1./(Wx)), where W is a n*p matrix.
%% hessian_f(x) := WTdiag(1/(Wx).^2)W.
%% lambda       := (y - x0)'*hessian_f(x0)*(y - x0).
%%

Wx0     = W*x0;
inv_Wx0 = 1./Wx0;
grad    = -(inv_Wx0'*W)';

% solve the sub problem
d = inv_Wx0.^2;
[y, info] = IVMQuadSimplex_away_port(grad - ((d.*Wx0)'*W)', W, d, y0, theta, max_iter);
WymWx0 = W*y - Wx0;
lambda = sqrt((d.*WymWx0)'*WymWx0);

end