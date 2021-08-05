function [Sol,i]    = PortFWBS(Fval, a, b, tol, n)

Vl                  = Fval(a);
Vr                  = Fval(b);
Sol                 = 2/(n+1);
for i               = 1: 5000
    if abs(Vl)      <= tol
        Sol         = a;
%         fprintf( 'Binary search stops after %2d iterations\n', i);
        break;
    end
    
    if abs(Vr)      <= tol
        Sol         = b;
%         fprintf( 'Binary search stops after %2d iterations\n', i);
        break;
    end
    c               = (a+b)/2;
    Vc              = Fval(c);
    if Vc * Vl      > 0
        a           = c;
        Vl          = Vc;
    else
        b           = c;
        Vr          = Vc;
    end
end

end