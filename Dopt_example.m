% Solve D-optimal design problem:
%
%   minimize -log(det(sum(AXAt)))
%   subj.to  x in Simplex
%   where A is a n*m matrix.
%
% The dual of this problem is called the minimum-volume enclosing 
% ellispsoid (MVEE) problem. Will compare our algorithm FWPN with a
% specialized Frank Wolfe (FW) away algorithm.
%
% References
% [1] Damla, S. A., Sun, P., and Todd, M. J. Linear convergence of a 
%     modified Frank–Wolfe algorithm for computing minimum-volume enclosing
%     ellipsoids. Optimisation Methods and Software, 23(1):5–19, 2008.

%% Set Path
clear;
addpath(genpath(pwd));

%% Generate Data
rng(123);
n = 1e2;
p = 1e3;

mu    = zeros(n,1);
sigma = rand(n,n);
sigma = sigma'*sigma;
A = mvnrnd(mu,sigma,p)';
At = A';

%% Solving the problem by using FW away
fprintf('************************************************************************\n')
fprintf('********** Solving D-optimal design problem by using FW-away ***********\n')
fprintf('************************************************************************\n')
param.rel_err = 1e-10;
if p <= 1e3
    param.maxit = 5e+3;
elseif p <= 1e4
    param.maxit = 2e+4;
end
param.break = 1;
param.disp = 1000;
hist_FW = DoptFWawaySolver( A, 1/p*ones(p,1), param);

%% Solving the problem by using our method
fprintf('************************************************************************\n')
fprintf('****** Solving D-optimal design problem by using our method(FWPN) ******\n')
fprintf('************************************************************************\n')
x0 = 1/p*ones(p,1);
Options.lambda0 = 10;
Options.lambda_tol = 1e-7; 
Options.sub_tol    = 0.01;
Options.short2long = 10;
Options.max_iter   = 100;
Options.sub_max_iter = 10*size(A,2);
get_obj   = @(x) DoptGetObj(x, A, At);
SubSolver = @(x, y, tol, max_iter) DoptFWPNSubSolver(x, y, A, At, tol, max_iter);
hist_FWPN = ProxNSolver(x0, Options, SubSolver, get_obj);                       

%% Plot the result
f_star = min([hist_FWPN.obj, hist_FW.obj]);
legend_fig1 = {};
MarkSize = 7;

semilogy([0, hist_FWPN.cumul_time], abs([hist_FWPN.f, hist_FWPN.obj] - f_star),...
    'o--', 'Color', [0.8500 0.3250 0.0980],...
    'MarkerEdgeColor',[0.8500 0.3250 0.0980], 'MarkerFaceColor',[0.8500 0.3250 0.0980], 'MarkerSize', MarkSize);  hold on
legend_fig1{1,1} = 'FWPN';
    
totallength = length(hist_FW.f);
dist = floor(totallength/10);
index = 1:dist:totallength;
semilogy(hist_FW.cumul_time(index), abs(hist_FW.f(index) - f_star),...
    'd-','color', [0.4660 0.6740 0.1880],...
    'MarkerEdgeColor',[0.4660 0.6740 0.1880], 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'MarkerSize', MarkSize);  hold on
legend_fig1{1,2} = 'FW-away';

xlabel('Time($s$)', 'Interpreter', 'latex', 'FontSize', 20);

ylabel('$f(x) - f^\star$','Interpreter', 'latex', 'FontSize', 20);

title(['$n$ = ', num2str(n), ', $p$ = ', num2str(p)],'Interpreter', 'latex', 'FontSize', 20)
h1 = legend(legend_fig1); 
set(h1, 'Interpreter', 'latex', 'FontSize', 12);