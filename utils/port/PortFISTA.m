function x_nxt  = PortFISTA(Grad, Hopr, x, L, tol)
y               = x;
x_cur           = y;
t               = 1;
kmax            = 1200;

for k           = 1: kmax
    DQ          = Hopr(y-x) + Grad;
    x_tmp       = y - 1/L * DQ;
    x_nxt       = PortProxSplx(x_tmp);     
    xdiff       = x_nxt - x_cur;
    ndiff       = norm(xdiff);
    if (ndiff < tol) && (k > 1)
%         fprintf('Fista err = %3.3e; Subiter = %3d; subproblem converged!\n', ndiff, k);
        break;
    end
    t_nxt       = 0.5 * (1+sqrt( 1+4*t^2 ));
    y           = x_nxt + (t-1)/t_nxt * xdiff;
    
    t           = t_nxt;
    x_cur       = x_nxt;
end

end