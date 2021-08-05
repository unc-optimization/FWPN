% Purpose: This is an implementation of the nonmonotne spectral gradient
% descent method for the logistic regression problem.

function output = LogRegnSPG(A, y, muf, rho, x0, options)
    
        
    % Initialization.
    gamma = options.gamma;
    alpha = options.alpha;
    Alpha = options.Alpha;
    M = options.M;
    sigma = options.sigma;
    tol = options.tol;
    maxiters = options.maxiters;
    printdist = options.printdist;
    n = size(y,1);
    
    time1 = tic;
    x_cur = x0;
    f_candi = Inf + zeros(M, 1);
    rel_gap = Inf;
    fprintf(' Iter  |  Obj Val  |  Rel Gap  |  Time \n');
    
    % The main loop.
    for iter = 1:maxiters
        
        % Evaluate the objective gradient and value.
        Ax    = (A*x_cur) .* y;
        expAx = exp(-Ax);
        px  = 1./(1+expAx)-1;
        ypx = y.*px;
        grad_fx_cur  = (ypx' * A)'/n + muf*x_cur;
        fx_val = sum(log(1+expAx))/n + muf*sum(x_cur.*x_cur)/2;
        output.f(iter) = fx_val;
        
        % Build f_candi
        f_candi = cat(1, f_candi(2:end), fx_val);
        
        % Decide the step-size.
        f_max = max(f_candi);
        lambda = alpha;
        x_next  = ProjectOntoL1Ball(x_cur - lambda*grad_fx_cur, rho);
        Ax    = (A*x_next) .* y;
        expAx = exp(-Ax);
        fx_next = sum(log(1+expAx))/n + muf*sum(x_next.*x_next)/2;
        inn_iter = 1;
        
        while fx_next > f_max + gamma*((x_next - x_cur)'*grad_fx_cur) && inn_iter < 1000
                lambda = sigma*lambda;
                x_next  = ProjectOntoL1Ball(x_cur - lambda*grad_fx_cur, rho);
                Ax    = (A*x_next) .* y;
                expAx = exp(-Ax);
                fx_next = sum(log(1+expAx))/n + muf*sum(x_next.*x_next)/2;
                inn_iter = inn_iter + 1;
        end
        
        % Update alpha
        x_diff = x_next - x_cur;
        px  = 1./(1+expAx)-1;
        ypx = y.*px;
        grad_fx_next  = (ypx' * A)'/n + muf*x_next;
        grad_diff = grad_fx_next - grad_fx_cur;
        alpha = max(Alpha(1), min(Alpha(2), ...
            (x_diff'*grad_diff)/(grad_diff'*grad_diff) ));
        output.f(iter) = fx_next;
        output.cumul_time(iter) = toc(time1);
        
        
        % Print and store the iteration.
        if mod(iter,printdist)==0
            fprintf(' %5d | %3.3e | %3.3e | %3.3e \n', iter, output.f(iter), rel_gap, output.cumul_time(iter));
        end
                
        % compute the relative gaps.
        rel_gap = norm(x_cur-ProjectOntoL1Ball(x_cur-grad_fx_cur, rho));
        
        % Check the stopping criterion.
        
        if rel_gap <= tol && iter > 1
            output.msg = 'Convergence achieved';
            break;
        end

        % Move to the next loop.
        x_cur = x_next;
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