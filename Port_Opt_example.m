% Solve the portfolio optimization problem:
%
%    min f(x) = -sum(log < w_i ,x > ) 
%    s.t. x in Simplex
%    where x in R^p, w_i in R^p, i = 1,...,n.
%
% Define W = [w_1T;...;w_nT] which is a n*p matrix, we can rewrite the 
% problem as:
%
%    min f(x) = -eTlog(Wx)
%    s.t. x in Simplex.
%
%    gradient_f(x) = -WT(1./(Wx)).
%    hessian_f(x) = WTdiag(1/(Wx).^2)W.
%
% Will compare our algorithm FWPN with Proximal Newton (PN),
% Frank Wolfe (FW), Frank Wolfe with Line Search (FW-LS) and
% Proximal Gradient with Barzilai-Brorwein's step-size (PG-BB)
%
% References
% [1] Frank,M.andWolfe,P. Analgorithmforquadraticprogramming. 
%     Naval Research Logistics Quarterly, 3:95–110, 1956.
% [2] Jaggi, M. Revisiting Frank-Wolfe: Projection-Free Sparse
%     Convex Optimization. JMLR W&CP, 28(1):427–435, 2013.
%% Set Path
clear;
addpath(genpath(pwd));

%% Choosing the data
use_real_data = 1;
if use_real_data == 1
    id = 1;
    plist = {'473500_wk.mat','625723_wk.mat','625889_wk.mat'};
    pname = plist{id};
    load(pname);
    [n,p] = size(W);
    fprintf(' We are solving real problem of size n is %5d and p is %5d.\n', n,p);
    fprintf('\n');
else
    n = 1e+4;
    p = 1e+3;
    W = PortGenData(n, p, 0.1);
    fprintf(' We are solving synthetic problem of size n is %5d and p is %5d.\n', n,p);
    fprintf('\n');
end
    
%% Preprocessing the data    
if (min(min(W)) >= 0)
    fprintf('Data is valid.\n');
    fprintf('\n');
else
    fprintf('Data is adjusted by taking exp.\n');
    fprintf('\n');
    W = exp(W);
end
x0    = ones(p,1)/p;
    
%% Solving the problem by using standard Proximal Newton method
fprintf('************************************************************************\n')
fprintf('*************** Solving portfolio problem by using PN ******************\n')
fprintf('************************************************************************\n')
options.Miter       = 1000;
options.printst     = 1;
tols.main           = 1e-8;
options.Lest        = 1;
hist_PN = PortPNSolver(W, x0, options, tols);

%% Solving the problem by using proximal-gradient method with B-B stepsize
options.cpr         = 0;
options.Miter       = 10000;
options.printst     = 100;
tols.main           = 1e-6;
fprintf('************************************************************************\n')
fprintf('************** Solving portfolio problem by using PG-BB ****************\n')
fprintf('************************************************************************\n')
hist_PGBB = PortPGBBSolver(W, x0, options, tols);

%% Solving the problem by using Frank-Wolfe method
options.cpr         = 0;
options.linesearch  = 0;
options.Miter       = 10000;
options.printst     = 1000;
tols.main           = 1e-4;
fprintf('************************************************************************\n')
fprintf('*************** Solving portfolio problem by using FW ******************\n')
fprintf('************************************************************************\n')
hist_FW = PortFWSolver(W, x0, options, tols);

%% Solving the problem by using Frank-Wolfe method with line search
options.cpr         = 0;
options.linesearch  = 1;
options.Miter       = 10000;
options.printst     = 1000;
tols.main           = 1e-4;
fprintf('************************************************************************\n')
fprintf('************** Solving portfolio problem by using FW-LS ****************\n')
fprintf('************************************************************************\n')
hist_FWLS = PortFWSolver(W, x0, options, tols);

%% Solving the problem by using our method
Options.lambda0    = 1;
Options.lambda_tol = 1e-4;
Options.sub_tol    = 0.1;
Options.short2long = 10;
Options.max_iter   = 100;
Options.sub_max_iter = size(W,2);
get_obj   = @(x) -sum(log(W*x));
SubSolver = @(x, y, tol, max_iter) PortFWPNSubSolver(x, y, W, tol, max_iter);
fprintf('************************************************************************\n')
fprintf('********** Solving portfolio problem by using our method(FWPN) *********\n')
fprintf('************************************************************************\n')
hist_FWPN = ProxNSolver(x0, Options, SubSolver, get_obj);

%% Plot the result
figure;

f_star = min([hist_FWPN.obj, hist_FW.f(end), hist_PN.f(end), hist_FWLS.f(end), hist_PGBB.f(end)]);
legend_fig1 = {};
MarkSize = 7;

semilogy([0, hist_FWPN.cumul_time], abs([hist_FWPN.f, hist_FWPN.obj] - f_star),...
    'o--', 'Color', [0.8500 0.3250 0.0980],...
    'MarkerEdgeColor',[0.8500 0.3250 0.0980], 'MarkerFaceColor',[0.8500 0.3250 0.0980], 'MarkerSize', MarkSize);  hold on
legend_fig1{1,1} = 'FWPN';

semilogy(hist_PN.cumul_time(1:end-1), abs(hist_PN.f - f_star),...
    '^-', 'Color', [0.9290 0.6940 0.1250],...
    'MarkerEdgeColor',[0.9290 0.6940 0.1250], 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerSize', MarkSize);  hold on
legend_fig1{1,2} = 'PN';
    
totallength = length(hist_FW.f);
dist = floor(totallength/10);
index = 1:dist:totallength;
semilogy(hist_FW.cumul_time(index), abs(hist_FW.f(index) - f_star),...
    'd-','color', [0.4660 0.6740 0.1880],...
    'MarkerEdgeColor',[0.4660 0.6740 0.1880], 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', MarkSize);  hold on
legend_fig1{1,3} = 'FW';

totallength = length(hist_FWLS.f);
dist = floor(totallength/10);
index = 1:dist:totallength;
semilogy(hist_FWLS.cumul_time(index), abs(hist_FWLS.f(index) - f_star),...
    'v-','color', [0 0.4470 0.7410],...
    'MarkerEdgeColor',[0 0.4470 0.7410], 'MarkerFaceColor', [0 0.4470 0.7410], 'MarkerSize', MarkSize);  hold on
legend_fig1{1,4} = 'FWLS';

totallength = length(hist_PGBB.f);
dist = floor(totallength/10);
index = 1:dist:totallength;
semilogy(hist_PGBB.cumul_time(index), abs(hist_PGBB.f(index) - f_star),...
    's-', 'Color',[0.3010 0.7450 0.9330],...
    'MarkerEdgeColor', [0.3010 0.7450 0.9330], 'MarkerFaceColor',[0.3010 0.7450 0.9330], 'MarkerSize', MarkSize);  hold on
legend_fig1{1,5} = 'PGBB';

xlabel('Time($s$)', 'Interpreter', 'latex', 'FontSize', 20);

ylabel('$f(X) - f^\star$','Interpreter', 'latex', 'FontSize', 20);

if use_real_data == 1
    title('real: $n = $' + string(n) + ', $p = $' + string(p),'Interpreter', 'latex', 'FontSize', 20)
else
        title('syn: $n = $' + string(n) + ', $p = $' + string(p),'Interpreter', 'latex', 'FontSize', 20)
end
h1 = legend(legend_fig1);
xlim([-0.3,inf]);
set(h1, 'Interpreter', 'latex', 'FontSize', 12);