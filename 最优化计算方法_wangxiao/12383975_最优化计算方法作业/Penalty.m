clc
clear

f = @(t) t(1)+t(2);%目标函数
c = @(t) t(1)^2 + t(2)^2 - 2;%目标函数的约束函数
Ce = [c];%根据题目等式约束

alpha = 1;
sigma0 = 10;
eps = 1e-5;
x0 = [-10,10]';%初始点选在了可行域外

[xk,k] = penalty(f,Ce,x0,sigma0,alpha,eps);


function [xk,k] = penalty(f,Ce,x0,sigma0,alpha,eps)
P = @(x) sum(Ce(x).^alpha);
sigma = sigma0;
xk = x0;
k = 0;
c = 2;
while abs(sigma*P(xk))>= eps
F = @(x) f(x)+ sigma*P(x);

xk = fminunc(F,xk);
sigma = c*sigma;
k = k+1
xk
end

end
