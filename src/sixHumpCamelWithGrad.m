

function [obj, grad] = sixHumpCamelWithGrad(x)
    obj = sixHumpCamel(x);
    if nargout>1
        grad = sixHumpCamelGrad(x);
    end
end

function obj = sixHumpCamel(x)
    obj = (4-2.1*x(1)^2+x(1)^4/3)*x(1)^2+x(1)*x(2)+(-4+4*x(2)^2)*x(2)^2;
end

function grad = sixHumpCamelGrad(x)
    grad = zeros(2, 1);
    grad(1) = x(2)+8.0*x(1)-8.4*x(1)^3+2*x(1)^5;
    grad(2) = x(1)-8.0*x(2)+16.0*x(2)^3;
end
