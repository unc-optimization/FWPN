% Purpose: This is an implementation of the accelerated proximal gradient
%          method with linesearch and restart for composite convex
%          optimization problems.

function output = LogRegAPG_LSRS_Solver(fx, gradfx, gx, proxg, ...
    Lf, x0, maxiters, tol, lambda)

% Check the inputs.
if nargin < 9,        lambda   = 1e-1;   end
if nargin < 8,        tol      = 1.0e-3; end
if nargin < 7,        maxiters = 500;    end
if isempty(tol),      tol      = 1.0e-3; end
if isempty(maxiters), maxiters = 500;    end
if nargin < 6, error('It requires at least 6 inputs!'); end

% Initialization.
lsmit = 10;
time1 = tic;
y_cur = x0;
x_cur = x0;
t_cur = 1;
L     = Lf;
LS    = 1;
fprintf(' Iter  |  Obj Val  |  Rel Gap  |  Time\n');
        
% The main loop.
for iter = 1:maxiters
    
    % Evaluate the objective gradient and value.
    grad_fx_cur = gradfx(y_cur);
    fx_val      = fx(x_cur) + gx(x_cur);
    output.f(iter) = fx_val;
    
    % Update the next iteration.
    fy_val      = fx(y_cur);
    nrm_dfx     = norm(grad_fx_cur, 2);
    if LS == 1
        L           = max(Lf/8, L/4);
        for itls = 1:lsmit
            x_next  = y_cur - (1/L)*grad_fx_cur;
            fx_val_new = fx(x_next);
            if fx_val_new <= fy_val - (0.5/L)*nrm_dfx.^2
                break;
            end
            L       = 1.5*L;
            %if L > Lf, L = Lf; x_next  = y_cur - (1/L)*grad_fx_cur; end
        end
    else
        L = 2*Lf;
    end
    
    % Perform the proximal gradient step.
    x_next  = proxg( y_cur - (1/L)*grad_fx_cur, 1/L );
    
    % Update t_cur and y_cur.
    t_next  = 0.5*(1 + sqrt(1 + 4*t_cur^2));
    y_next  = x_next + ((t_cur-1)/t_next)*(x_next - x_cur);
    output.cumul_time(iter+1) = toc(time1);
    
    % compute the relative gaps.
    grad_fx = gradfx(x_cur);
    rel_gap = norm(x_cur-proxg(x_cur-grad_fx, lambda));
    
    % Check the stopping criterion.
    if rel_gap <= tol && iter > 1
        output.msg = 'Convergence achieved';
        break;
    end
    
    % Print and store the iteration.
    if mod(iter,200)==1
        fprintf(' %5d | %3.3e | %3.3e | %3.3e \n', iter, fx_val, rel_gap, output.cumul_time(iter+1));
        output.hist.fx(iter,1)      = fx_val;
        output.hist.rel_gap(iter,1) = rel_gap;
    end
    
    
    % Restart ...
    if (x_next - x_cur)'*(y_cur - x_next) > 0
        y_next = x_next;
        t_next = 1;
    end
    
    % Move to the next loop.
    x_cur = x_next;
    y_cur = y_next;
    t_cur = t_next;
end
% End of the main loop.

% Finalization.
if iter >= maxiters
    output.msg = 'Exceed the maximum number of iterations';
end
output.xopt    = x_cur;
output.iter    = iter;
output.time    = toc(time1);
output.f       = fx_val;
output.rel_gap = rel_gap;
end