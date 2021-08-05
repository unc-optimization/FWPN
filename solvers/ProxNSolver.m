function output = ProxNSolver(x0, Options, SubSolver, get_obj, k)
% Proximal-Newton method 
% Solve : min f(x) + g(x)
%         where f is self-concordance and g could be a non-smooth function.

% set k
if nargin == 4
    k = 1;
end

% Get parameters
lambda0    = Options.lambda0;
lambda_tol = Options.lambda_tol;
sub_tol    = Options.sub_tol;
short2long   = Options.short2long;
max_iter     = Options.max_iter;
sub_max_iter = Options.sub_max_iter;

% Initialization
count_time = 0;
x_cur = x0;
y_cur = x0;
lambda_cur = lambda0;
fprintf(" step type | Iter |  lambda   |  Obj Val   | Sub Iter| Sub Rel Err |  Sub Gap  |  Time \n");
    
for iter = 1:max_iter
    tic;
    
    % Solve Subproblem
    [y_cur, lambda_next, f_cur, sub] = SubSolver(x_cur, y_cur,...
        min((sub_tol*lambda_cur)^k, 1), sub_max_iter);%min(max(100/lambda_cur,100),sub_max_iter)
    
    count_time = count_time + toc;
    
    % Choose Long or Damped Newton Step
    if lambda_next <= short2long
        x_next = y_cur;
        step_info = 'long step';
    else
        x_next = x_cur + 1/(1+lambda_next)*(y_cur - x_cur);
        step_info = 'short step';
    end
    
    % Print result
    fprintf("%10s |  %3d | %3.3e | %3.3e |  %5d  |  %3.3e  | %3.3e | %3.3e\n",...
        step_info, iter, lambda_next, f_cur, sub.iter, sub.rel_err, sub.gap, count_time);
    output.f(iter) = f_cur;
    output.cumul_time(iter) = count_time;
    
    % Check the stop criterion
    if lambda_next <= lambda_tol %||(iter > 10 && lambda_next > 1.2*lambda_cur)
        break;
    end
    
    % Move to the next iteration
    x_cur = x_next;
    lambda_cur = lambda_next;
    
end


%% Get Output
output.iter = iter;
output.time = count_time;
output.obj  = get_obj(x_next);
output.xopt = x_next;
end