function [y, lambda, info] = LogRegIVMSubSolver(x0, y0, A, tempt, C, muf, theta, max_iter)
%% Conditional Gradient Method to solve the subproblem of logistic regression with elastic net:
%%
%%      min  <grad_f(x0), u> + 1/2< hessian_f(x0)*(u - x0), u - x0>
%%      s.t. ||u||_1 <= rho1.
%%
%% lambda := (y - x0)'*hessian_f(x0)*(y - x0).
%%

% %% method 1
% n = size(A,1);
% 
% Ax0      = A*x0;
% temp     = 1 + exp(Ax0);
% inv_temp = 1./temp;
% temp1    = 1/n*(1 - inv_temp); 
% grad     = (temp1'*A)' + muf*x0;
% obj_val  = 1/n*sum(log(temp)) + muf*sum(x0.^2)/2 + rho*sum(abs(x0));
% 
% % solve the sub problem
% d = temp1.*inv_temp;
% [y, info] = QuadSimplex_logistic(grad - ((d.*Ax0)'*A)' - muf*x0, A, d, muf, rho1, y0, tol, max_iter);
% delta = y-x0;
% AymAx0 = A*y - Ax0;
% lambda = sqrt((d.*AymAx0)'*AymAx0 + muf*sum(delta.^2));

% %% method 2
% % seperate x by x1 - x2
% [n, p] = size(A);
% 
% Ax0      = A*x0;
% temp     = 1 + exp(Ax0);
% inv_temp = 1./temp;
% temp1    = 1/n*(1 - inv_temp); 
% grad     = (temp1'*A)' + muf*x0;
% obj_val  = 1/n*sum(log(temp)) + muf*sum(x0.^2)/2 + rho*sum(abs(x0));
% 
% % solve the sub problem
% d = temp1.*inv_temp;
% d_hat   = rho1^2*d;
% muf_hat = rho1^2*muf;
% q       = grad - ((d.*Ax0)'*A)' - muf*x0;
% q_hat   = rho1*q;
% 
% res = 1 - sum(abs(y0));
% y0 = [max(y0, 0); max(-y0,0)] + res/2/p;
% [y, info] = QuadSimplex_logistic_simplex(q_hat, A, d_hat, muf_hat, y0, tol, 2*max_iter);
% 
% % back to the original space
% y = y(1:p) - y(p+1:end);
% delta = y-x0;
% AymAx0 = A*y - Ax0;
% lambda = sqrt((d.*AymAx0)'*AymAx0 + muf*sum(delta.^2));

%% method 3
n = size(A,1);

CtC = C'*C;
Ax0      = A*x0;
temp     = 1 + exp(Ax0);
inv_temp = 1./temp;
temp1    = 1/n*(1 - inv_temp);
grad     = (temp1'*A)' + muf*CtC*x0;

% solve the sub problem
d = temp1.*inv_temp;
[y, info] = IVMQuadSimplex_away_logreg(grad - ((d.*Ax0)'*A)' - muf*CtC*x0, A, tempt, d, muf*CtC, y0, theta, max_iter);
temp = C*(y-x0);
AymAx0 = A*y - Ax0;
lambda = sqrt((d.*AymAx0)'*AymAx0 + muf*(temp'*temp));

end
