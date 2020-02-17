function hist    = PortPGBBSolver(W, x, options, tols)

n                       = size(W,1);
p                       = size(W,2);
hist.nsize              = [n,p];

if (isfield(options,'cpr')   == 0)
    options.cpr     = 0;         % no comparison is made
end

time1                   = tic;
hist.cumul_time(1)      = 0;
fprintf(' Iter  |  Obj Val   | Step Size | Rel Diff  |   Time \n');

for iter                = 1:options.Miter    
    
    % Compute the denominator.
    denom               = W * x;                % n by 1 vector
    
    % Evaluate the gradient.
    ratG                = 1 ./ denom;           % n by 1 vector
    Grad                = -W' * ratG;           % p by 1 vector
    
    if iter             == 1
        nume            = W * Grad;
        Gradtau         = @(t)( sum(nume ./ (denom - t*nume)));
        [tau, ~]        = PortFWBS(Gradtau,0,1,1e-7, iter);
    else
        Gradif          = Grad - Grad_pre;
        tau             = (diffx' * Gradif) / (Gradif' * Gradif); 
    end
    x_nxt_tmp           = x - tau * Grad;
    x_nxt               = PortProxSplx(x_nxt_tmp);
    
    diffx               = x_nxt - x;
    
    % solution value stop-criterion    
    nrm_dx              = norm(diffx);
    rdiff               = nrm_dx / max(1.0, norm(x)); 
    hist.err(iter, 1)   = rdiff;
    hist.f(iter)        = -sum(log(denom));
    hist.cumul_time(iter+1) = toc(time1);
    
    % Check the stopping criterion.
    if (rdiff           <= tols.main) && iter > 1
        hist.msg        = 'Convergence achieved!';
        fprintf(' %4d  | %3.3e | %3.3e | %3.3e | %3.3e\n',...
            iter, hist.f(iter), tau, rdiff, hist.cumul_time(iter+1));
        break;
    end
    
    x                   = x_nxt;
    Grad_pre            = Grad;
    
    if mod(iter, options.printst) == 0 || iter == 1
        fprintf(' %4d  | %3.3e | %3.3e | %3.3e | %3.3e\n',...
            iter, hist.f(iter), tau, rdiff, hist.cumul_time(iter+1));
    end
    
    % Check the comparison stop criterion if needed
    if options.cpr      == 1
        Val             = -sum(log( W*x ));
        if abs(Val - options.val)/max(1, options.val) <= tols.cpr
            fprintf('Objective Value achieved the accuracy!\n');
            break;
        end
    end
  
end

% if mod(iter, options.printst) ~= 0
%     fprintf('iter = %4d, stepsize = %3.3e, rdiff = %3.3e\n', iter, s, rdiff);
% end

if iter                 >= options.Miter
    hist.msg            = 'Exceed the maximum number of iterations';
end

% Output and Records
hist.time               = toc(time1);
hist.iter               = iter;
hist.xopt               = x;
hist.obj                = -sum(log( W*x ));
end