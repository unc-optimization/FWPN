function output = LogRegPGSolver(data,muf,rho,x0,param)
%% ************************************************************************
%%
%% Attempt: Using proximal gradient method to solve logistic regression.
%% 
%% Input : 
%%          data  : A cell that consists of {(y_i,a_i) | i = 1,2,...,n}
%%          muf   : parameter of l2-norm, positive
%%          rho   : l1 penalized parameter, positive
%%          x0    : initial point
%%          param : user define for algorithm
%%
%% Output:  
%%          xsol  : optimal solution
%%          output: output infomation

%% ************************************************************************

p      = size(data.X,2);

% check inputs.
if nargin < 5
    param.maxiter  = 10000;
    param.tol      = 1e-6;
    param.printyes = 1;
end

if isempty(param)
    param.maxiter  = 10000;
    param.tol      = 1e-6;
    param.printyes = 1;
end

if nargin < 4, x0   = zeros(p,1); end
if isempty(x0), x0  = zeros(p,1); end
if nargin < 3, rho  = 0.001;      end
if isempty(rho),rho = 0.001;      end
if nargin < 2, muf  = 1.0;        end
if isempty(muf),muf = 1.0;        end

% initialization.
x        = x0;
breakyes = 0;
time1    = tic;
output.cumul_time(1) = 0;
fprintf(' Iter |  Obj Val  |  Rel Gap  |  Time\n');

% compute the Lipschitz constant for stepsize.
Lip = compute_Lip(data,muf);

% main loop.
for iter = 1:param.maxiter

    % evaluate the objective funcition and gradient
    [fx,g]    = logistic(data,x);
    g         = muf*x + g;
    output.f(iter) = fx + muf*sum(x.*x)/2;
    
    % compute the vector to do the proximal operator.
    step = 2/Lip;
    u    = x - step*g;
    
    % perform the proximal operator.
    x  = ProjectOntoL1Ball(u, rho);
    output.cumul_time(iter+1) = toc(time1);
    
    % compute the relative gap.
    %relgap(iter) = abs(rho*sum(abs(x)) + sum(g.*x));
    output.relgap(iter) = norm(x-ProjectOntoL1Ball(x-g, rho));
       
    % check for termination.
    if output.relgap(iter) < param.tol
        breakyes   = 1;
        output.msg = 'Converged.';
    end
    
    if param.printyes && mod(iter, param.printdist) == 0
        fprintf('%5d | %3.3e | %3.3e | %3.3e \n',...
            iter,output.f(iter),output.relgap(iter),output.cumul_time(iter+1));
    end
    
    if breakyes
        break;
    end
    
    
end
% end of main loop.

% update output info.
output.iter   = iter;
output.Lip    = Lip;
output.xopt   = x;
output.time   = toc(time1);
output.obj    = logistic(data, x) + muf*sum(x.*x)/2;
end

%% ************************************************************************
%  compute Lipschitz constant of f.
function Lip = compute_Lip(data,muf)

n = size(data.y,1);
A = transpose(data.X);
A = sparse(A);
l2nrm = DecoptNormAtAeval('numeric', A, 20, 1e-5, A');
Lip   = l2nrm/4/n + muf;
end

%% ************************************************************************
% logistic function.
function [fx,gx] = logistic(data,x)
y = data.y;
A = sparse(data.X);
n = size(y,1);

Ax    = (A*x) .* y;
expAx = exp(-Ax);

% evaluate the function values.
fx = sum(log(1+expAx))/n;

% evaluate the gradient.
px  = 1./(1+expAx)-1;
ypx = y.*px;
gx  = (ypx' * A)'/n;
end