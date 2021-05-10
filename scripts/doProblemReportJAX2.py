import time
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import jax.config
import autograd.numpy as anp
import autograd

jax.config.update("jax_enable_x64", True)

def rosenbrock_jax(x):
    answer = jnp.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

def rosenbrock_autograd(x):
    answer = anp.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

jEvalFunc = jax.jit(rosenbrock_jax)
jGradFunc= jax.jit(jax.grad(rosenbrock_jax))
aEvalFunc = rosenbrock_autograd
aGradFunc = autograd.grad(rosenbrock_autograd)

x0 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e4]
x1 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e2]
toleranceChange = 1e-9
maxIter = 10000
jx0 = jnp.array(x0)
jx1 = jnp.array(x1)
ax0 = anp.array(x0)
ax1 = anp.array(x1)

jOptimRes_x0 = scipy.optimize.minimize(fun=jEvalFunc, x0=jx0, method='BFGS', jac=jGradFunc)
startTime = time.time()
jOptimRes_x1 = scipy.optimize.minimize(fun=jEvalFunc, x0=jx1, method='BFGS', jac=jGradFunc)
jElapsedTime = time.time()-startTime

j2OptimRes_x0 = jax.scipy.optimize.minimize(fun=jEvalFunc, x0=jx0, method='BFGS')
startTime = time.time()
j2OptimRes_x1 = jax.scipy.optimize.minimize(fun=jEvalFunc, x0=jx1, method='BFGS')
j2ElapsedTime = time.time()-startTime

aOptimRes_x0 = scipy.optimize.minimize(fun=aEvalFunc, x0=ax0, method='BFGS', jac=aGradFunc)
startTime = time.time()
aOptimRes_x1 = scipy.optimize.minimize(fun=aEvalFunc, x0=ax1, method='BFGS', jac=aGradFunc)
aElapsedTime = time.time()-startTime

print("scipy.optimize and JAX grads converged with x0?: {}".format(jOptimRes_x0.fun<1e-6))
print("scipy.optimize and JAX grads converged with x1?: {}".format(jOptimRes_x1.fun<1e-6))
print("scipy.optimize and JAX grads elapsed time with x1: {:f} sec".format(jElapsedTime))
print("")
print("jax.scipy.optimize and JAX grads converged with x0?: {}".format(j2OptimRes_x0.fun<1e-6))
print("jax.scipy.optimize and JAX grads converged with x1?: {}".format(j2OptimRes_x1.fun<1e-6))
print("jax.scipy.optimize and JAX grads elapsed time with x1: {:f} sec".format(j2ElapsedTime))
print("")
print("scipy.optimize and Autograd grads converged with x0?: {}".format(aOptimRes_x0.fun<1e-6))
print("scipy.optimize and Autograd gards converged with x1?: {}".format(aOptimRes_x1.fun<1e-6))
print("scipy.optimize and Autograd grads elapsed time with x1: {:f} sec".format(aElapsedTime))

import pdb
pdb.set_trace()
