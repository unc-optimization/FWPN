function output = DoptFWawaySolver( A, x0, param)   % A, b, m, l, lambda, param)
% Frank Wolfe(Away) Algorithm to solve D-optimal design problem
%
%       Problem :
%           minimize    -  log(det(sum(xi*Mi))) = -  log(det(AXAt))
%       	subj.to     x in Simplex.
%
%           f(x)        =   -  log(det(AXAt))
%           grad_f(x)   = [wi]_i := -[ait*(AXAt)^(-1)*ai]_i;
%
%       FW Update:
%           ej = argmin_s  {  <s, grad_f(x)> | s in Simplex}
%           x^(k+1) = x^k + t_k*(ej - x^k)
%           t_k = (-wj/n - 1)/(-wj - 1)
%
%       FW Away Update:
%           ei = argmax_s  {  <s, grad_f(x)> | s in Simplex}
%           x^(k+1) = x^k + t_k*(ei - x^k)
%           t_k = max{(-wi/n - 1)/(-wi - 1), xi/(xi - 1)} if -wi > 1
%               = xi/(xi - 1)                             else

%% Frank-Wolfe

% options
if ~isfield(param,'maxit'),             param.maxit = 2e4;                      end
if ~isfield(param,'disp'),              param.disp = ceil(param.maxit/100);     end

% performance counters
countTime = 0;  % Cumulative time

% Initialization
[n,m] = size(A);
x_cur  =   x0;
grad_f_cur = zeros(m,1);
WholeIndex = 1:m;
mask  = (x_cur > 0);

% Get gradient and hessian
AXAt = A*sparse(1:m,1:m,x_cur,m,m)*A';
R = chol(AXAt);
Rinv = inv(R);
invAXAt = Rinv*Rinv';
invAXAt = (invAXAt + invAXAt')/2;
for i = 1:m
    temp = A(:,i);
    grad_f_cur(i) = -temp'*invAXAt*temp;
end
f_cur  = -sum(log(diag(R).^2));
output.f(1) = f_cur;

fprintf(' Iter |   Obj Val  |  Rel Err  |    Gap    |  Time\n');
for iter=2:param.maxit
    
    % Solve Subproblem
    tic;
    
    % FW update
    [Floor, indexFW] = min(grad_f_cur); % linear oracle
    gapFW = -n - Floor;
    
    % Away update
    ActCoeff = grad_f_cur(mask);
    ActIndex = WholeIndex(mask);
    [Ceil, indexAway] = max(ActCoeff); % linear oracle
    gapAway = Ceil - (-n);
    indexAway = ActIndex(indexAway);
    
    % Check use FW or Away
    if gapFW >= gapAway
        
        % Use FW update
        alpha = (-grad_f_cur(indexFW)/n - 1)/(-grad_f_cur(indexFW) - 1);
        z = zeros(m,1); z(indexFW) = 1;
        delta = z - x_cur;
        x_next = x_cur + alpha*delta;
        f_next = f_cur - log((1-alpha)^(n-1)*(1 - alpha - alpha*grad_f_cur(indexFW)));
        
        % Update (AXAt)^(-1)
        temp = invAXAt*A(:,indexFW);
        temp1 = (temp'*A)';
        invAXAt = 1/(1-alpha)*(invAXAt - ...
                  alpha/(1 - alpha - alpha*grad_f_cur(indexFW))...
                  *(temp*temp'));
        grad_f_cur = 1/(1-alpha)*(grad_f_cur + ...
                 alpha/(1 - alpha - alpha*grad_f_cur(indexFW))...
                 *(temp1.^2));
    else
        
        % Use Away Update
        if -grad_f_cur(indexAway) > 1

            temp1 = x_cur(indexAway)/(x_cur(indexAway) - 1);
            temp2 = (-grad_f_cur(indexAway)/n - 1)/(-grad_f_cur(indexAway) - 1);
            alpha = max(temp1, temp2);
        else
            
            alpha = x_cur(indexAway)/(x_cur(indexAway) - 1);
        end
        
        z = zeros(m,1); z(indexAway)  = 1;
        delta = z - x_cur;
        x_next = x_cur + alpha*delta;
        x_next = max(x_next, 0);
        f_next = f_cur - log((1-alpha)^(n-1)*(1 - alpha - alpha*grad_f_cur(indexAway)));
        
        % Update (AXAt)^(-1)
        temp = invAXAt*A(:,indexAway);
        temp1 = (temp'*A)';
        invAXAt = 1/(1-alpha)*(invAXAt - ...
                  alpha/(1 - alpha - alpha*grad_f_cur(indexAway))...
                  *(temp*temp'));
        grad_f_cur = 1/(1-alpha)*(grad_f_cur + ...
                 alpha/(1 - alpha - alpha*grad_f_cur(indexAway))...
                 *(temp1.^2));
    end
    
    % check the stop criterion
    rel_err = abs(f_next - f_cur)/max(abs(f_cur),1);
    if rel_err <= param.rel_err && param.break
        break;
    end
    
    % move to the next iteration
%         for i = 1:m
%             temp = A(:,i);
%             grad_f_cur(i) = -temp'*invAXAt*temp;
%         end
    x_cur = x_next;
    f_cur = f_next;
    mask = (x_cur > 0);
    
    countTime = countTime + toc;
    
    % Save history
    output.f(iter) = f_next;
    output.gapFW(iter-1) = gapFW;
    output.gapAway(iter-1) = gapAway;
    output.cumul_time(iter-1) = countTime;
    
    % Print
    if mod(iter,param.disp)==0
        fprintf('%6d| %3.3e | %3.3e | %3.3e | %3.2e \n', ...
            iter, f_next, rel_err, gapFW, countTime);
    end
    
end

output.xopt = x_next;
output.obj  = f_next;
output.time = countTime;
output.iter = iter;

end