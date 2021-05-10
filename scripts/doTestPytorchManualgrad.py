import sys
import pdb
import time
import os
import copy
import torch
import numpy as np
import argparse
import configparser
sys.path.append("../src")
import testFunctions
import utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("paramsConfigFilename", help="parameters configuration filename")
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-1)
    parser.add_argument("--resultsFilenamePattern", help="results filename pattern", default="../results/pytorch_manualgrad_{:s}_lr{:f}.csv")
    args = parser.parse_args()

    paramsConfigFilename = args.paramsConfigFilename
    lr = float(args.lr)
    resultsFilenamePattern = args.resultsFilenamePattern

    resultsDesc, _ = os.path.splitext(os.path.basename(paramsConfigFilename))
    resultsFilename = resultsFilenamePattern.format(resultsDesc, lr)

    paramsConfig = configparser.ConfigParser()
    paramsConfig.read(paramsConfigFilename)
    generativeFunc = paramsConfig["data_params"]["generativeFunc"]
    minima = np.array([float(str) for str in paramsConfig["data_params"]["minima"][1:-1].split(",")])
    x0 = torch.tensor([float(str) for str in paramsConfig["init_params"]["x0"][1:-1].split(",")], dtype=torch.double)
    minima = minima.reshape((-1, len(x0)))
    maxIter = int(paramsConfig["optim_params"]["maxIter"])
    toleranceGrad = float(paramsConfig["optim_params"]["toleranceGrad"])
    toleranceChange = float(paramsConfig["optim_params"]["toleranceChange"])
    lineSearchFn = paramsConfig["optim_params"]["lineSearchFn"]
    nRepeats = int(paramsConfig["test_params"]["nRepeats"])

    if generativeFunc=="rosenbrock":
        evalFunc = testFunctions.rosenbrock_pytorch
        evalGrad = testFunctions.rosenbrockManualgrad
    elif generativeFunc=="sixHumpCamel":
        evalFunc = testFunctions.sixHumpCamel
        evalGrad = testFunctions.sixHumpCamelManualgrad
    elif generativeFunc=="zakharov":
        evalFunc = testFunctions.zakharov_pytorch
        evalGrad = testFunctions.zakharovManualgrad_pytorch
    else:
        raise ValueError("Unrecognized generativeFunc={:s}".format(generativeFunc))

    curEval = torch.tensor([-float("inf")])

    def closure():
        nonlocal curEval

        curEval = evalFunc(x=x[0])
        x[0].grad = torch.tensor(evalGrad(x=x[0]))
        return curEval

    results = torch.zeros((nRepeats, 2))
    print("x0=", x0)
    for i in range(nRepeats):
        x = [copy.deepcopy(x0)]
        optimizer = torch.optim.LBFGS(x, max_iter=maxIter, lr=lr, line_search_fn=lineSearchFn, tolerance_grad=toleranceGrad, tolerance_change=toleranceChange)
        startTime = time.time()
        optimizer.step(closure)
        elapsedTime = time.time()-startTime
        results[i,0] = utils.minL2NormToMinima(x=x[0].detach().numpy(), minima=minima)
        results[i,1] = elapsedTime
    np.savetxt(resultsFilename, results.detach().numpy(), delimiter=",")
    # pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
