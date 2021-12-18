clc
clear

f = @(t)t(1)^2+2*t(2)^2;
x0 = [1,1]';
epsilon = 1e-5;
H0 = eye(2);
method = 'BFGS';%DFP 此处可以根据需要变换算法
x = quasi_Newton(f,x0,epsilon,H0,method);

function [xk,k] = quasi_Newton(f,x0,epsilon,H0,method)
%使用：quasi_Newton(f,x0,method)

if nargin < 3
    help mfilename;
end
k = 0;
syms t1 t2;
t = [t1,t2]';
fs = f(t);
dfs = gradient(fs);
df = matlabFunction(dfs);
df = @(x) df(x(1),x(2));
df0 = df(x0);
normdf = sqrt(df0'*df0);
H = H0;
xk = x0;
dfk = df0;
while normdf > epsilon
    p = -H*dfk;
    alpha = cal_alpha(H,dfk,p);
    xk1 = xk + alpha*p;
    dfk1 = df(xk1);
    sk = xk1 - xk;
    yk = dfk1 - dfk;
    eval(['H = ' method '(H,sk,yk);']);
    %H = BFGS(H,sk,yk);
    k = k + 1;
    xk = xk1;
    dfk = dfk1;
    normdf = sqrt(dfk'*dfk);
    xk
    f(xk)
end
end
function alpha = cal_alpha(H,dfk,p)
    Q = [2,0;0,4];
    alpha = -dot(dfk,p) / dot(p, Q*p);
    %alpha = dfk'*H*dfk/(dfk'*H'*dfk); 
    %alpha = 1;
end
function H = BFGS(H,sk,yk)%BFGS算法代码
    gammak = 1/(yk'*sk);
    skykT = sk * yk';
    skskT = sk * sk';
    E = eye(2);
    H = (E-gammak*skykT)*H*(E-gammak*skykT') + gammak*skskT;
end
function H = DFP(H,sk,yk)%DFP算法代码
    Hyk = H*yk;
    ykTsk =  yk'*sk;
    skskT = sk * sk';
    H = H - (Hyk*yk'*H)/(yk'*Hyk) + skskT/ykTsk;
end