import time
import scipy.optimize
import jax.numpy as jnp
import autograd.numpy as anp
import jax
import autograd

def rosenbrock(x):
    answer = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

jEvalFunc = jax.jit(rosenbrock)
jGradFunc= jax.jit(jax.grad(rosenbrock))
aEvalFunc = rosenbrock
aGradFunc = autograd.grad(rosenbrock)

x0 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e4]
x1 = [76.0, 97.0, 20.0, 120.0, 0.01, 1e2]
jx0 = jnp.array(x0)
jx1 = jnp.array(x1)
ax0 = anp.array(x0)
ax1 = anp.array(x1)

jOptimRes_x0 = scipy.optimize.minimize(fun=jEvalFunc, x0=jx0, method='BFGS', jac=jGradFunc)
startTime = time.time()
jOptimRes_x1 = scipy.optimize.minimize(fun=jEvalFunc, x0=jx1, method='BFGS', jac=jGradFunc)
jElapsedTime = time.time()-startTime

aOptimRes_x0 = scipy.optimize.minimize(fun=aEvalFunc, x0=ax0, method='BFGS', jac=aGradFunc)
startTime = time.time()
aOptimRes_x1 = scipy.optimize.minimize(fun=aEvalFunc, x0=ax1, method='BFGS', jac=aGradFunc)
aElapsedTime = time.time()-startTime

print("JAX converged with x0?: {}".format(jOptimRes_x0.fun<1e-6))
print("JAX converged with x1?: {}".format(jOptimRes_x1.fun<1e-6))
print("JAX elapsed time with x1: {:0.2f} sec".format(jElapsedTime))
print("")
print("Autograd converged with x0?: {}".format(aOptimRes_x0.fun<1e-6))
print("Autograd converged with x1?: {}".format(aOptimRes_x1.fun<1e-6))
print("Autograd lapsed time with x1: {:0.2f} sec".format(aElapsedTime))

import pdb
pdb.set_trace()
