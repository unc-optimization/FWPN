% Purpose: This is an implementation of the accelerated proximal gradient 
%          method with restart for composite convex optimization problems.

function output = LogRegAPG_RS_Solver(fx, gradfx, gx, proxg, Lf, ...
                                             x0, maxiters, tol, printdist)
    
    % Check the inputs.
    if nargin < 8,        tol      = 1.0e-3; end
    if nargin < 7,        maxiters = 500;    end
    if isempty(tol),      tol      = 1.0e-3; end
    if isempty(maxiters), maxiters = 500;    end
    if nargin < 6, error('It requires at least 6 inputs!'); end
        
    % Initialization.
    time1 = tic;
    y_cur = x0;
    x_cur = x0;
    t_cur = 1;
    L     = Lf;
    rel_gap = Inf;
    output.cumul_time(1) = 0;
    fprintf(' Iter  |  Obj Val  |  Rel Gap  |  Time \n');
    
    % The main loop.
    for iter = 1:maxiters
        
        % Evaluate the objective gradient and value.
        grad_fx_cur = gradfx(y_cur);
        fx_val      = fx(x_cur) + gx(x_cur);
        output.f(iter) = fx_val;
        
        % Update the next iteration.
        x_next  = proxg( y_cur - (1/L)*grad_fx_cur, 1/L);
        t_next  = 0.5*(1 + sqrt(1 + 4*t_cur^2));
        y_next  = x_next + ((t_cur-1)/t_next)*(x_next - x_cur);
        output.cumul_time(iter+1) = toc(time1);
        
        
        % Print and store the iteration.
        if mod(iter,printdist)==0
            fprintf(' %5d | %3.3e | %3.3e | %3.3e \n', iter, fx_val, rel_gap, output.cumul_time(iter+1));
        end
                
        % compute the relative gaps.
        grad_fx = gradfx(x_cur);
        rel_gap = norm(x_cur-proxg(x_cur-grad_fx, 1));
        
        % Check the stopping criterion.
        
        if rel_gap <= tol && iter > 1
            output.msg = 'Convergence achieved';
            break;
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
    output.obj     = fx_val;
    output.rel_gap = rel_gap;
end