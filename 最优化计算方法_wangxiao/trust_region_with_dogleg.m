[~,~] = trust_region()

function [x,k] = trust_region()
f = @(x1,x2)100*(x2-x1^2)^2 + (1-x1)^2; 
syms x1 x2; 
f_x = f(x1,x2); 

x0 = [0;0]; %初始点
Delta_hat = 10; %delta最大上限，人为设定
Delta0 = 0.1; %delta初值，人为设定
eta = 0;  %eta，0到1/4，人为设定
eps = 1e-10; %判断||pk|| == delta_k，人为设定

df = diff_handle(f_x); %梯度
B = hessian(f_x);  %hessian矩阵
Delta_k = Delta0; 
x = x0; 
step = 20; %一共的迭代次数
for k = 0:step-1
    fk = f(x(1),x(2)); 
    dfk = df(x(1),x(2)); 
    Bk = subs(B,{x1,x2},{x(1),x(2)}); 
    Bk = double(Bk); %不同步时代入x1,x2后的hessian
    pk = cal_pk(dfk,Bk,Delta_k);  %dogleg求解pk子问题
    m = @(p) fk + dfk'*p + 1/2*p'*Bk*p; 
    rho_k = cal_rho_k(f,m,x,pk);  %计算rho_k,评估信赖域大小
    if rho_k < 1/4 
       Delta_k = 1/4*Delta_k; 
    elseif rho_k > 3/4 && abs(norm(pk,2) - Delta_k)< eps
       Delta_k = min(2*Delta_k, Delta_hat);
    else
       Delta_k = Delta_k;
    end
    
    if rho_k > eta
        x = x + pk;
    else
        x = x;
    end
    
    if k <= 1 %(显示前两次迭代的结果)
        fprintf('第%d 次迭代后 x 的值为：\n',k+1);  %对结果进行打印输出 
        disp(x) 
    end
    fprintf('第%d 次迭代后 y 值为：%f \n',k+1, f(x(1),x(2)));
end 
end

function df = diff_handle(f_s) 
syms x1 x2; 
df = [diff(f_s,x1); diff(f_s,x2)]; 
df = matlabFunction(df); 
end 

function tau = cal_tau(pB,pU,Delta) 
npB = sqrt(pB'*pB); 
npU = sqrt(pU'*pU); 
if npB <= Delta %完全在域内
    tau = 2; 
elseif npU >= Delta %交点在第一条
    tau = Delta/npU; 
else %交点在第二条
    a = dot(pB,pB) - 2 * dot(pB,pU) + dot(pU,pU);
    b = 2 * dot(pB, pU) - 2 * dot(pU,pU);
    c = dot(pU, pU) - Delta^2;
    tau = (-1 * b + sqrt(b^2 - 4 * a *c)) / (2 * a);
    tau = tau + 1; 
end
end

function pk = cal_pk(dfk,Bk,Delta) 
pU = -dfk' * dfk /(dfk' * Bk * dfk) * dfk;
pB = -Bk^(-1)*dfk; 
tau = cal_tau(pB,pU,Delta); 
if tau >=0 && tau <=1 
    pk = tau*pU; 
elseif tau >= 1 && tau <=2 
    pk = pU + (tau-1)*(pB-pU); 
else
    error('tau 的值不能为%f',tau); 
end

end

function rho_k = cal_rho_k(f,m,x,pk) 
rho_k = (f(x(1),x(2)) - f(x(1)+pk(1),x(2)+pk(2)))/((m([0;0])-m(pk))); 
end 
