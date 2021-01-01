import sys
import pdb
import time
import os
import numpy as np
import autograd.numpy as anp
import scipy.optimize
import autograd
import argparse
import configparser
sys.path.append("../src")
import testFunctions
import utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("paramsConfigFilename", help="parameters configuration filename")
    parser.add_argument("--resultsFilenamePattern", help="results filename pattern", default="../results/scipy_HIPSautograd_{:s}.csv")
    args = parser.parse_args()

    paramsConfigFilename = args.paramsConfigFilename
    resultsFilenamePattern = args.resultsFilenamePattern

    resultsDesc, _ = os.path.splitext(os.path.basename(paramsConfigFilename))
    resultsFilename = resultsFilenamePattern.format(resultsDesc)

    paramsConfig = configparser.ConfigParser()
    paramsConfig.read(paramsConfigFilename)
    generativeFunc = paramsConfig["data_params"]["generativeFunc"]
    minima = anp.array([float(str) for str in paramsConfig["data_params"]["minima"][1:-1].split(",")])
    x0 = anp.array([float(str) for str in paramsConfig["init_params"]["x0"][1:-1].split(",")])
    minima = minima.reshape((-1, len(x0)))
    maxIter = int(paramsConfig["optim_params"]["maxIter"])
    toleranceGrad = float(paramsConfig["optim_params"]["toleranceGrad"])
    toleranceChange = float(paramsConfig["optim_params"]["toleranceChange"])
    nRepeats = int(paramsConfig["test_params"]["nRepeats"])

    if generativeFunc=="rosenbrock":
        evalFunc = testFunctions.rosenbrock
    elif generativeFunc=="sixHumpCamel":
        evalFunc = testFunctions.sixHumpCamel
    elif generativeFunc=="zakharov":
        evalFunc = testFunctions.zakharov_autogradNumpy
    else:
        raise ValueError("Unrecognized generativeFunc={:s}".format(generativeFunc))

    gradFunc = autograd.grad(evalFunc)

    minimizeOptions = {'ftol': toleranceChange, 'gtol': toleranceGrad, 'maxiter': maxIter}

    results = anp.zeros((nRepeats, 2))
    print("x0=", x0)
    for i in range(nRepeats):
        startTime = time.time()
        optimRes = scipy.optimize.minimize(fun=evalFunc, x0=x0, method='L-BFGS-B', jac=gradFunc, options=minimizeOptions)
        elapsedTime = time.time()-startTime
        results[i,0] = utils.minL2NormToMinima(x=optimRes.x, minima=minima)
        results[i,1] = elapsedTime
    np.savetxt(resultsFilename, results, delimiter=",")
    # pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
