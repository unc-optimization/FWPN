function [y, test] = QuadSimplex(q, Q, y0, tol, max_iter)
% This function Use Conditional Gradient Away Method to solve problem of
%   min 1/2*(x'*Q*x) + q'*x
%   s.t. x in Simplex

y = y0;
mask  = (y > 0);
n = size(Q, 1);
WholeIndex = 1:n;

% Get current coefficient
Qy  = Q*y;
Grad = q + Qy;
test_obj_cur = q'*y + 1/2*(y'*Qy);

for iter = 1: max_iter
    
    % Solve Subproblem
    % FW update
    lin_obj = Grad'*y;
    [floor, indexFW] = min(Grad); % linear oracle
    gapFW = lin_obj - floor;
    % Away update
    ActCoeff = Grad(mask);
    ActIndex = WholeIndex(mask);
    [ceil, indexAway] = max(ActCoeff); % linear oracle
    gapAway = ceil - lin_obj;
    indexAway = ActIndex(indexAway);
    
    % Damped step
    % Check use FW or Away
    if gapFW >= gapAway
        
        % Use FW update
        Qz = Q(:,indexFW);
        Qz_y = Qz - Qy;
        deltaFW = -y; deltaFW(indexFW) = 1 - y(indexFW);
        alpha = gapFW/(deltaFW'*Qz_y);
        alpha = min(1, alpha);
        y = (1-alpha)*y;
        y(indexFW) = y(indexFW) + alpha;
        Qy = Qy + alpha*Qz_y;
    else
        
        % Use Away Update
        Qz = Q(:,indexAway);
        Hessiany_z = Qy - Qz;
        deltaAway = y; deltaAway(indexAway) = y(indexAway) - 1;
        alpha = gapAway/(deltaAway'*Hessiany_z);
        AlphaMax = y(indexAway)/(1 - y(indexAway));
        alpha = min(AlphaMax, alpha);
        y = (1 + alpha)*y;
        y(indexAway) = y(indexAway) - alpha;
        Qy = Qy + alpha*Hessiany_z;
    end

    % Save History
    test_obj(iter) = test_obj_cur;
    test_gapFW(iter) = gapFW;
    test_gapAway(iter) = gapAway;
    test_obj_next = q'*y + 1/2*(y'*Qy);
    test_rel_err = abs(test_obj_next - test_obj_cur)/max(abs(test_obj_cur), 1);
    
    % Check the stop criterion
    if gapFW <= tol && gapAway <= tol
        break;
    end
    
    % Move to the next iteration
    Grad = q + Qy;
    test_obj_cur = test_obj_next;
    mask = (y > 0);
end

test.iter = iter;
test.rel_err = test_rel_err;
test.gap = gapFW;

end

