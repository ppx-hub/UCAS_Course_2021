[~,~] = trust_region()

function [x,k] = trust_region()
f = @(x1,x2)100*(x2-x1^2)^2 + (1-x1)^2; 
syms x1 x2; 
f_x = f(x1,x2); 

x0 = [0;0]; %��ʼ��
Delta_hat = 10; %delta������ޣ���Ϊ�趨
Delta0 = 0.1; %delta��ֵ����Ϊ�趨
eta = 0;  %eta��0��1/4����Ϊ�趨
eps = 1e-10; %�ж�||pk|| == delta_k����Ϊ�趨

df = diff_handle(f_x); %�ݶ�
B = hessian(f_x);  %hessian����
Delta_k = Delta0; 
x = x0; 
step = 20; %һ���ĵ�������
for k = 0:step-1
    fk = f(x(1),x(2)); 
    dfk = df(x(1),x(2)); 
    Bk = subs(B,{x1,x2},{x(1),x(2)}); 
    Bk = double(Bk); %��ͬ��ʱ����x1,x2���hessian
    pk = cal_pk(dfk,Bk,Delta_k);  %dogleg���pk������
    m = @(p) fk + dfk'*p + 1/2*p'*Bk*p; 
    rho_k = cal_rho_k(f,m,x,pk);  %����rho_k,�����������С
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
    
    if k <= 1 %(��ʾǰ���ε����Ľ��)
        fprintf('��%d �ε����� x ��ֵΪ��\n',k+1);  %�Խ�����д�ӡ��� 
        disp(x) 
    end
    fprintf('��%d �ε����� y ֵΪ��%f \n',k+1, f(x(1),x(2)));
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
if npB <= Delta %��ȫ������
    tau = 2; 
elseif npU >= Delta %�����ڵ�һ��
    tau = Delta/npU; 
else %�����ڵڶ���
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
    error('tau ��ֵ����Ϊ%f',tau); 
end

end

function rho_k = cal_rho_k(f,m,x,pk) 
rho_k = (f(x(1),x(2)) - f(x(1)+pk(1),x(2)+pk(2)))/((m([0;0])-m(pk))); 
end 
