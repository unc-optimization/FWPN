function hist       = PortPNSolver(W, x, options, tols)

if (isfield(tols,'steps')   == 0)
    tols.steps      = 0.05;         % How small is step-size to break damped step
end

if (isfield(tols,'sub')    == 0)
    tols.sub        = 0.25*tols.main;         % Terminate Condition for Subproblem
end

n                   = size(W,1);
p                   = size(W,2);
hist.nsize          = [n,p];

s_deactive              = 0;
time1                   = tic;
hist.cumul_time(1)      = 0;
fprintf(' Iter  |  Obj Val   | Step Size | Rel Diff  |   Time \n');

if options.Lest     == 1                    % Estimate Lipschitz Constant
    Denom           = W * ones(p,1)/p;
    RatH            = 1 ./ (Denom.^2);    
    dir             = ones(p,1);
    for Liter       = 1:15
        Dir         = W'* (RatH .* (W * dir));
        dir         = Dir / norm(Dir);
    end
    Hd              = W'* (RatH .* (W * dir));
    dHd             = dir' * Hd;
    L               = dHd / ( dir' * dir );
end

for iter                = 1:options.Miter    
    
    % Compute the denominator.
    denom               = W * x;                % n by 1 vector
    
    % Evaluate the gradient.
    ratG                = 1 ./ denom;           % n by 1 vector
    Grad                = -W' * ratG;           % p by 1 vector
    
    % Evaluate the Hessian
    ratH                = 1 ./ (denom.^2);      % n by 1 vector
    Hopr                = @(d)( W'* (ratH .* (W * d) ) );
                                                % p by 1 vector
 
    if options.Lest     == 0                    % Compute Lipschitz Constant  
        dir                 = ones(p,1);
        for Liter           = 1:20
            Dir             = Hopr(dir);
            dir             = Dir / norm(Dir);
        end
        Hd                  = Hopr(dir);
        dHd                 = dir' * Hd;
        L                   = dHd / ( dir' * dir );
    end
    
    x_nxt               = PortFISTA(Grad, Hopr, x, L, tols.sub);
    
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
        fprintf(' %4d  | %3.3e | %3.3e | %3.3e | %3.3e \n',...
            iter, hist.f(iter), s, rdiff, hist.cumul_time(iter+1));
        break;
    end
    
     % Compute a step-size if required.
    if ~s_deactive
        Hd              = Hopr(diffx);
        dHd             = diffx' * Hd;
        lams            = sqrt(dHd);
        s               = 1 / (1 + lams);   % 0.5 * Mf = 1
    end
    if (1-s <= tols.steps), s = 1; s_deactive = 1; end                    
    x                   = x + s * diffx;
    
    if mod(iter, options.printst) == 0 || iter == 1
        fprintf(' %4d  | %3.3e | %3.3e | %3.3e | %3.3e \n',...
            iter, hist.f(iter), s, rdiff, hist.cumul_time(iter+1));
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