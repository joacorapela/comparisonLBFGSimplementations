import time
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import jax.config
import autograd.numpy as anp
import autograd

jax.config.update("jax_enable_x64", True)

def zakharov_jaxNumpy(x):
    def zakharovFromIndices(x, ii):
        sum1 = jnp.sum(jnp.power(x, 2))
        sum2 = jnp.sum(0.5*ii*x)
        answer = sum1+sum2**2+sum2**4
        return answer

    ii = jnp.arange(1, len(x)+1, step=1)
    answer = zakharovFromIndices(x=x, ii=ii)
    return answer

def zakharov_autogradNumpy(x):
    def zakharovFromIndices(x, ii):
        sum1 = (x**2).sum()
        sum2 = (0.5*ii*x).sum()
        answer = sum1+sum2**2+sum2**4
        return answer

    ii = anp.arange(1, len(x)+1, step=1)
    answer = zakharovFromIndices(x=x, ii=ii)
    return answer

jEvalFunc = jax.jit(zakharov_jaxNumpy)
aEvalFunc = zakharov_autogradNumpy
aGradFunc = autograd.grad(aEvalFunc)

x0 = [600.0, 700.0, 200.0, 100.0, 90.0, 1e4]
toleranceChange = 1e-9
maxIter = 10000
jx0 = jnp.array(x0)
ax0 = anp.array(x0)

startTime = time.time()
aOptimRes_x0 = scipy.optimize.minimize(fun=aEvalFunc, x0=ax0, method='BFGS', jac=aGradFunc)
aElapsedTime = time.time()-startTime

startTime = time.time()
jOptimRes_x0 = jax.scipy.optimize.minimize(fun=jEvalFunc, x0=jx0, method='BFGS')
jElapsedTime = time.time()-startTime

print("scipy.optimize converged? {}, elapsed time: {}".format(aOptimRes_x0.fun<1e-6, aElapsedTime))
print("jax.scipy.optimize converged? {}, elapsed time: {}".format(jOptimRes_x0.fun<1e-6, jElapsedTime))

import pdb
pdb.set_trace()
