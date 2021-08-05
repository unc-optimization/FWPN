function hist = PortFWSolver(W, x, options, tols)

n                       = size(W,1);
p                       = size(W,2);
hist.nsize              = [n,p];
nsearch                 = 0;

if (isfield(options,'cpr')   == 0)
    options.cpr     = 0;         % no comparison is made
end

time1                   = tic;
fprintf(' Iter  |  Obj Val   | Step Size | Rel Diff  | Search Num |   Time \n');

for iter                = 1:options.Miter    
    
    % Compute the denominator.
    denom               = W * x;                % n by 1 vector
    
    % Evaluate the gradient.
    ratG                = 1 ./ denom;           % n by 1 vector
    Grad                = -W' * ratG;           % p by 1 vector
    
    smin                = find(Grad == min(Grad));
    s                   = double(1:p == smin)';
    
    if options.linesearch == 0
        tau             = 2/(iter+1);           % since our k starts from 1
    else
        nume            = W * (s-x);
        Gradtau         = @(t)( sum(nume ./ (denom + t*nume)));
        [tau,nsearch]   = PortFWBS(Gradtau, 0,1, tols.main*0.1, iter);
    end
    x_nxt               = (1-tau) * x + tau * s;            
    
    diffx               = x_nxt - x;
    
    x                   = x_nxt;
    
    % solution value stop-criterion    
    nrm_dx              = norm(diffx);
    rdiff               = nrm_dx / max(1.0, norm(x)); 
    hist.err(iter, 1)   = rdiff;
    hist.f(iter)        = -sum(log(W*x));
    hist.cumul_time(iter) = toc(time1);
    
    % Check the stopping criterion.
    if (rdiff           <= tols.main) && iter > 1
        hist.msg        = 'Convergence achieved!';
        fprintf(' %5d | %3.3e | %3.3e | %3.3e |     %2d     | %3.3e\n', ...
            iter, hist.f(iter), tau, rdiff, nsearch, hist.cumul_time(iter));
        break;
    end
    
    
    if mod(iter, options.printst) == 0 || iter == 1
        fprintf(' %5d | %3.3e | %3.3e | %3.3e |     %2d     | %3.3e\n', ...
            iter, hist.f(iter), tau, rdiff, nsearch, hist.cumul_time(iter));
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

time1                   = toc(time1);
if iter                 >= options.Miter
    hist.msg            = 'Exceed the maximum number of iterations';
end

% Output and Records
hist.time               = time1;
hist.iter               = iter;
hist.xopt               = x;
hist.obj                = -sum(log( W*x ));

end