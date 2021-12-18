CG_method()

function CG_method() 
A = [4 1 
     1 3];            
b=-[1 2]'; 
x0=[2 1]';  %初始值 
max_iter=1000;  %最大的迭代次数 
[y,iter]=cgm(A,b,x0,max_iter); 
fprintf('迭代次数:\n   %d \n',iter); 
fprintf('=======================\n')
fprintf('方程的解: \n'); 
fprintf('%.6f\n',y); 
fprintf('解的值: \n'); 
fprintf('%.6f\n',1/2 * y' * A * y + b' * y); 

end 

function [x_k,iter] = cgm (A,b,x0,max_iter) 
x_k=x0; 
eps=1.0e-6; 
fprintf('\n x0 = '); 
fprintf('%10.6f\n',x0); 
r_k = b + A * x_k; %这里是r0，初值
p_k = -r_k; %这里是p0，初值
for k=0:max_iter 
    alpha_k=(r_k'*r_k)/(p_k'*A*p_k); 
    x_k1=x_k+alpha_k*p_k; 
    r_k1 = r_k + alpha_k * A * p_k;
    beta=(r_k1'*r_k1)/(r_k'*r_k);    
    p_k=-r_k1+beta*p_k; %新的p_k就是p_k+1
    x_k=x_k1; 
    r_k=r_k1; 
    if (norm(r_k1,2))<= eps
        iter = k+1; 
        x_k=x_k1; 
        r_k=r_k1; 
        fprintf('x%d = ',k+1); 
        fprintf('%10.6f\n',x_k); 
        return 
    end 
    fprintf('x%d = ',k+1);   
    fprintf('%10.6f\n',x_k); 
    end 
iter = max_iter;
return  
end 