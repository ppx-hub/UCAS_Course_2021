clc
clear

%% prepare f
ft = @(t)t(1)^2+t(2)^2+t(1)^4;
syms t1 t2;
f = ft([t1,t2]);
df = gradient(f);
ddf = hessian(f);
dft =matlabFunction(df,'vars',[t1,t2]);
dft = @(t)dft(t(1),t(2));
ddft = matlabFunction(ddf,'vars',[t1,t2]);
ddft = @(t)ddft(t(1),t(2));

%% init 
epsilon = 0.001;%0.1%0.001   %此处取0.1 0.01 0.001三种进行计算
x0 = [epsilon;epsilon];
alpha = 1;

%%
x_current = x0;
for i = 1:1
    B = ddft(x_current);
    p = -B\dft(x_current);
    x_current = x_current + alpha*p
end
scale_x = norm(x_current,2)
log(scale_x)./log(epsilon)
