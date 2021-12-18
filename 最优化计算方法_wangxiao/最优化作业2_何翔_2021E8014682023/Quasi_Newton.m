clc
clear

f = @(t)t(1)^2+100*t(2)^2;
x0 = [1,1]';
epsilon = 1e-5; %convergence tolerance
H0 = eye(2); %��Hessian����

method = 'BFGS';%��ѡBFGS����DFP

fprintf('ʹ�÷���Ϊ%s\n', method)

x = quasi_Newton(f,x0,epsilon,H0,method);

function [xk,k] = quasi_Newton(f,x0,epsilon,H0,method)
%ʹ�ã�quasi_Newton(f,x0,method)

k = 0;
syms t1 t2;
t = [t1,t2]';
fs = f(t);
dfs = gradient(fs);
df = matlabFunction(dfs);

df = @(x) df(x(1),x(2));
H = H0;
xk = x0;
dfk = df(x0);
normdf = norm(dfk,2);
while normdf > epsilon
    p = -H*dfk;
    alpha = Calalpha(dfk,p);
    xkk = xk + alpha*p;
    dfkk = df(xkk);
    sk = xkk - xk;
    yk = dfkk - dfk;
    if strcmp(method,'BFGS') == 1
        H = BFGS(H,sk,yk);
    else if strcmp(method,'DFP') == 1
        H = DFP(H,sk,yk);
        end
    end
    xk = xkk;
    dfk = dfkk;
    normdf = norm(dfk,2);
    fprintf('��%d �ε����� xk ��ֵΪ��\n',k+1);  %�Խ�����д�ӡ��� 
    disp(xk);
    fprintf('��%d �ε����� f(xk) ��ֵΪ��\n',k+1); 
    disp(f(xk));
    k = k + 1;
end
fprintf('----------------------------\n');  
fprintf('��%d �ε�����,������ֹ��������СֵΪ\n',k);  
disp(f(xk));
end
function alpha = Calalpha(dfk,p)
    Q = [2,0;0,200]; %f = 1/2X'QX
    alpha = -dot(dfk,p) / dot(p, Q*p); %���κ�����ȷ��������
end
function H = BFGS(H,sk,yk)%BFGS�㷨����
    rok = 1/(yk'*sk);
    skykT = sk * yk';
    skskT = sk * sk';
    I = eye(2);
    H = (I-rok*skykT)*H*(I-rok*skykT') + rok*skskT;
end
function H = DFP(H,sk,yk)%DFP�㷨����
    Hyk = H*yk;
    ykTsk =  yk'*sk;
    skskT = sk * sk';
    H = H - (Hyk*yk'*H)/(yk'*Hyk) + skskT/ykTsk; %��BFGS����H���¹�ʽ��ͬ
end