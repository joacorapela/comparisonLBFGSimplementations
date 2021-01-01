import pdb
import copy
import scipy.optimize
import torch

def rosenbrock(x):
    answer = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return answer

evalFunc = rosenbrock

x0s = [[76.0, 97.0, 20.0, 120.0, 0.01, 1e2], [76.0, 97.0, 20.0, 120.0, 0.01, 1e4]]
trueMins = [torch.ones(len(x0)) for x0 in x0s]

toleranceGrad = 1e-5
toleranceChange = 1e-9
xConvTol = 1e-4
lineSearchFn = "strong_wolfe"
maxIterOneEpoch = 1000
maxIterMultipleEpochs = 100
nEpochs = 10
assert maxIterOneEpoch==nEpochs*maxIterMultipleEpochs

def closure():
    optimizer.zero_grad()
    curEval = evalFunc(x=x[0])
    curEval.backward(retain_graph=True)
    return curEval

for x0, trueMin in zip(x0s, trueMins):
    print("Results for x0=", x0)

    xOneEpoch = torch.tensor(copy.deepcopy(x0))
    xOneEpoch.requires_grad = True
    x = [xOneEpoch]
    optimizer = torch.optim.LBFGS(x, max_iter=maxIterOneEpoch, line_search_fn=lineSearchFn, tolerance_grad=toleranceGrad, tolerance_change=toleranceChange)
    optimizer.step(closure)
    stateOneEpoch = optimizer.state[optimizer._params[0]]
    funcEvalsOneEpoch = stateOneEpoch["func_evals"]
    nIterOneEpoch = stateOneEpoch["n_iter"]
    print("\tResults for one epochs:")
    print("\t\tConverged: {}".format(torch.norm(xOneEpoch-trueMin, p=2)<xConvTol))
    print("\t\tFunction evaluations: {:d}".format(funcEvalsOneEpoch))
    print("\t\tIterations: {:d}\n".format(nIterOneEpoch))

    xMultipleEpochs = torch.tensor(copy.deepcopy(x0))
    xMultipleEpochs.requires_grad = True
    x = [xMultipleEpochs]
    optimizer = torch.optim.LBFGS(x, max_iter=maxIterMultipleEpochs, line_search_fn=lineSearchFn, tolerance_grad=toleranceGrad, tolerance_change=toleranceChange)
    for epoch in range(nEpochs):
        optimizer.step(closure)
    stateMultipleEpochs = optimizer.state[optimizer._params[0]]
    funcEvalsMultipleEpochs = stateMultipleEpochs["func_evals"]
    nIterMultipleEpochs = stateMultipleEpochs["n_iter"]
    print("\tResults for multiple epochs:")
    print("\t\tConverged: {}".format(torch.norm(xMultipleEpochs-trueMin, p=2)<xConvTol))
    print("\t\tFunction evaluations: {:d}".format(funcEvalsMultipleEpochs))
    print("\t\tIterations: {:d}\n".format(nIterMultipleEpochs))

pdb.set_trace()

