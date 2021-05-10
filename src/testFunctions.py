
import autograd.numpy as anp
import jax.numpy as jnp
import numpy as np
import torch

def zakharovFromIndices(x, ii):
    sum1 = (x**2).sum()
    sum2 = (0.5*ii*x).sum()
    answer = sum1+sum2**2+sum2**4
    return answer

def zakharov_pytorch(x):
    ii = torch.arange(1, len(x)+1, step=1)
    answer = zakharovFromIndices(x=x, ii=ii)
    return answer

def zakharov_numpy(x):
    ii = np.arange(1, len(x)+1, step=1)
    answer = zakharovFromIndices(x=x, ii=ii)
    return answer

def zakharov_autogradNumpy(x):
    ii = anp.arange(1, len(x)+1, step=1)
    answer = zakharovFromIndices(x=x, ii=ii)
    return answer

def zakharov_jaxNumpy(x):
    ii = jnp.arange(1, len(x)+1, step=1)
    answer = zakharovFromIndices(x=x, ii=ii)
    return answer

def zakharovManualgradFromIndices(x, ii):
    sum2 = (0.5*ii*x).sum()
    answer = 2*x+(sum2+2*sum2**3)*ii
    return answer

def zakharovManualgrad_pytorch(x):
    ii = torch.arange(1, len(x)+1, step=1)
    answer = zakharovManualgradFromIndices(x=x, ii=ii)
    return answer

def zakharovManualgrad_numpy(x):
    ii = np.arange(1, len(x)+1, step=1)
    answer = zakharovManualgradFromIndices(x=x, ii=ii)
    return answer

def zakharovWithAutograd(x):
    # type(x)==numpy.ndarray
    x = torch.from_numpy(x)
    x.requires_grad = True
    func = zakharov_pytorch(x)
    func.backward()
    func = func.detach().numpy()
    grad = x.grad.detach().numpy()

    return (func, grad)

def zakharovWithManualgrad(x):
    # type(x)==numpy.ndarray
    func = zakharov_numpy(x)
    grad = zakharovManualgrad_numpy(x)
    return (func, grad)

def sixHumpCamel(x):
    answer = (4-2.1*x[0]**2+x[0]**4/3)*x[0]**2+x[0]*x[1]+(-4+4*x[1]**2)*x[1]**2
    return answer

def sixHumpCamelManualgrad(x):
    grad = np.zeros_like(x)
    grad[0] = x[1]+8.0*x[0]-8.4*x[0]**3+2*x[0]**5
    grad[1] = x[0]-8.0*x[1]+16.0*x[1]**3
    return grad

def sixHumpCamelWithAutograd(x):
    x = torch.from_numpy(x)
    x.requires_grad = True
    func = sixHumpCamel(x)
    func.backward()
    func = func.detach().numpy()
    grad = x.grad.detach().numpy()

    return (func, grad)

def sixHumpCamelWithManualgrad(x):
    func = sixHumpCamel(x)
    grad = sixHumpCamelManualgrad(x)
    return (func, grad)

def rosenbrock_python(x):
    answer = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

def rosenbrock_pytorch(x):
    answer = torch.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

def rosenbrock_autogradNumpy(x):
    answer = anp.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

def rosenbrock_jaxNumpy(x):
    answer = jnp.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

def rosenbrockManualgrad(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    grad = np.zeros_like(x)
    grad[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    grad[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    grad[-1] = 200*(x[-1]-x[-2]**2)
    return grad

def rosenbrockWithAutograd(x):
    x = torch.from_numpy(x)
    x.requires_grad = True
    func = rosenbrock_pytorch(x)
    func.backward()
    func = func.detach().numpy()
    grad = x.grad.detach().numpy()

    return (func, grad)

def rosenbrockWithManualgrad(x):
    func = rosenbrock_python(x)
    grad = rosenbrockManualgrad(x)
    return (func, grad)

