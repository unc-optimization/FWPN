% Purpose: This is an implementation of the inexact variable metric
% method for constrained convex optimization problems.

function output = IVMSolver(x0, options, subSolver, get_obj, get_grad)
    
        
% Initialization.
M     = options.M;
tau = options.tau;
theta = options.theta;
maxiters     = options.maxiters;
lambda_tol   = options.lambda_tol;
sub_max_iter = options.sub_max_iter;

% Initialization
count_time = 0;
x_cur = x0;
y_cur = x0;
f_candi = Inf + zeros(M, 1);
% output.cumul_time(1) = 0;
% output.f(1) = get_obj(x_cur);
fprintf("Iter  |  lambda   |  Obj Val   | Sub Iter| Sub Rel Err |  Sub Gap  |  Time \n");

% The main loop.
for iter = 1:maxiters
    tic;

    % solve subproblem
    [y_cur, lambda_cur, sub] = subSolver(x_cur, y_cur, theta, sub_max_iter);
    
    % Build f_candi
    fx_cur = get_obj(x_cur);
    f_candi = cat(1, f_candi(2:end), fx_cur);

    % Decide the step-size.
    f_max = max(f_candi);
    alpha = 1;
    d_cur = y_cur - x_cur;
    x_next  = x_cur + alpha * d_cur;
    fx_next = get_obj(x_next);
    grad_fx_cur = get_grad(x_cur);
    
    
    while fx_next > f_max + tau*((x_next - x_cur)'*grad_fx_cur)
        alpha = alpha / 2;
        x_next = x_cur + alpha * d_cur;
        fx_next = get_obj(x_next);
    end
    count_time = count_time + toc;
    
    % Print result
    fprintf("  %3d | %3.3e | %3.3e |  %5d  |  %3.3e  | %3.3e | %3.3e\n",...
        iter, lambda_cur, fx_next, sub.iter, sub.rel_err, sub.gap, count_time);
    output.f(iter) = fx_next;
    output.cumul_time(iter) = count_time;
    
    % Check the stop criterion
    if lambda_cur <= lambda_tol
        break;
    end
    
    % Move to the next iteration
    x_cur = x_next;
end
% End of the main loop.

% Finalization.
if iter >= maxiters
    output.msg = 'Exceed the maximum number of iterations';
end
output.iter    = iter;
output.time    = count_time;
output.xopt    = x_cur;
output.obj     = get_obj(x_next);
end