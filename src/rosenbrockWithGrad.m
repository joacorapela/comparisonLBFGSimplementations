

function [obj, grad] = rosenbrockWithGrad(x)
    obj = rosenbrock(x);
    if nargout>1
        grad = rosenbrockGrad(x);
    end
end

function obj = rosenbrock(x)
    obj = sum(100.0*(x(2:end)-x(1:end-1).^2).^2 + (1-x(1:end-1)).^2);
end

function grad = rosenbrockGrad(x)
    xm = x(2:end-1);
    xm_m1 = x(1:end-2);
    xm_p1 = x(3:end);
    grad = zeros(size(x));
    grad(2:end-1) = 200*(xm-xm_m1.^2)-400*(xm_p1 - xm.^2).*xm-2*(1-xm);
    grad(1) = -400*x(1)*(x(2)-x(1).^2)-2*(1-x(1));
    grad(end) = 200*(x(end)-x(end-1).^2);
end
