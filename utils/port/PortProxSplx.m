function x      = fn_portf_proxsplx(y)

m               = length(y);
bget            = false;

s               = sort(y,'descend');
tmpsum          = 0;

for ii          = 1 : m-1
    tmpsum      = tmpsum + s(ii);
    tmax        = (tmpsum - 1)/ii;
    if tmax     >= s(ii+1)
        bget    = true;
        break;
    end
end
    
if ~bget, tmax = (tmpsum + s(m) -1)/m; end

x               = max( y-tmax, 0 );
return;
end