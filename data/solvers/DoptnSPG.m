% Purpose: This is an implementation of the nonmonotne spectral gradient
% descent method for the D-optimal design problem.

function output = DoptnSPG(A, At, x0, options)
    
        
    % Initialization.
    gamma = options.gamma;
    alpha = options.alpha;
    Alpha = options.Alpha;
    M = options.M;
    sigma = options.sigma;
    tol = options.tol;
    maxiters = options.maxiters;
    printdist = options.printdist;
    
    time1 = tic;
    x_cur = x0;
    f_candi = Inf + zeros(M, 1);
    rel_gap = Inf;
    fprintf(' Iter  |  Obj Val  |  Rel Gap  |  Time \n');
    
    % The main loop.
    for iter = 1:maxiters
        
        % Evaluate the objective gradient and value.
        grad_fx_cur = DoptnSPGGetGrad(x_cur, A, At);
        fx_val      = DoptnSPGGetObj(x_cur, A, At);
        output.f(iter) = fx_val;
        
        % Build f_candi
        f_candi = cat(1, f_candi(2:end), fx_val);
        
        % Decide the step-size.
        f_max = max(f_candi);
        lambda = alpha;
        x_next  = PortProxSplx(x_cur - lambda*grad_fx_cur);
        fx_next = DoptnSPGGetObj(x_next, A, At);
        inn_iter = 1;
        
        while fx_next > f_max + gamma*((x_next - x_cur)'*grad_fx_cur) && inn_iter < 1000
                lambda = sigma*lambda;
                x_next  = PortProxSplx(x_cur - lambda*grad_fx_cur);
                fx_next = DoptnSPGGetObj(x_next, A, At);
                inn_iter = inn_iter + 1;
        end
        
        % Update alpha
        x_diff = x_next - x_cur;
        grad_fx_next = DoptnSPGGetGrad(x_next, A, At);
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
        rel_gap = norm(x_cur-PortProxSplx(x_cur-grad_fx_cur));
        
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

function obj_val = DoptnSPGGetObj(x, A, At)

[n,m] = size(A);
Z = A*sparse(1:m,1:m,x,m,m)*At;

R = chol(Z + 0*eye(n));
obj_val = -sum(log(diag(R).^2));

end

function grad_val = DoptnSPGGetGrad(x, A, At)

[n,m] = size(A);
Z = A*sparse(1:m,1:m,x,m,m)*At;

R = chol(Z + 0*eye(n));
Rinv = inv(R);
InvZ = Rinv*Rinv';
InvZ = (InvZ+InvZ')/2;
AtInvZA  = At*InvZ*A;
grad_val = -diag(AtInvZA);
end