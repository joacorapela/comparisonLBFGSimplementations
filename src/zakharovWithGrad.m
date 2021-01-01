

function [obj, grad] = zakharovWithGrad(x)
    obj = zakharov(x);
    if nargout>1
        grad = zakharovGrad(x);
    end
end

function obj = zakharov(x)
    ii = 1:length(x);
    sum1 = sum(x.^2);
    sum2 = sum(0.5*ii*x);
    obj = sum1+sum2^2+sum2^4;
end

function grad = zakharovGrad(x)
    ii = (1:length(x))';
    sum2 = sum(0.5*ii.*x);
    grad = 2*x+(sum2+2*sum2^3)*ii;
end
