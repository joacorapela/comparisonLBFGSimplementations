
import sys
import pdb
import numpy as np
sys.path.append("../src")
import plotly.io as pio
import plotFunctions
import argparse

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor", help="results descriptor")
    parser.add_argument("learningRate", help="learning rate")
    parser.add_argument("--ylimElapsedTime", help="Limit for ordinates", default=None)
    parser.add_argument("--ylimApproxError", help="Limit for ordinates", default=None)
    parser.add_argument("--matlabResultsFilenamePattern", help="Matlab results filename pattern", default="../results/matlab_manualgrad_{:s}.csv")
    parser.add_argument("--pytorchResultsFilenamePattern", help="PyTorch results filename pattern", default="../results/pytorch_{:s}_{:s}_lr{:f}.csv")
    parser.add_argument("--scipyResultsFilenamePattern", help="SciPy results filename pattern", default="../results/scipy_{:s}_{:s}.csv")
    parser.add_argument("--jaxResultsFilenamePattern", help="JAX results filename pattern", default="../results/jax_{:s}.csv")
    parser.add_argument("--figFilenamePattern", help="Figure filename pattern", default="../figures/{:s}_{:s}_{:f}.{:s}")
    args = parser.parse_args()

    descriptor = args.descriptor
    lr = float(args.learningRate)
    ylimApproxError = args.ylimApproxError
    ylimElapsedTime = args.ylimElapsedTime
    matlabResultsFilenamePattern = args.matlabResultsFilenamePattern
    pytorchResultsFilenamePattern = args.pytorchResultsFilenamePattern
    scipyResultsFilenamePattern = args.scipyResultsFilenamePattern
    jaxResultsFilenamePattern = args.jaxResultsFilenamePattern
    figFilenamePattern = args.figFilenamePattern

    if ylimApproxError is not None:
        ylimApproxError = np.array([float(str) for str in ylimApproxError[1:-1].split(",")])

    if ylimElapsedTime is not None:
        ylimElapsedTime = np.array([float(str) for str in ylimElapsedTime[1:-1].split(",")])

    matlabResFilename = matlabResultsFilenamePattern.format(descriptor)
    pytorchAutogradResFilename = pytorchResultsFilenamePattern.format("autograd", descriptor, lr)
    pytorchManualgradResFilename = pytorchResultsFilenamePattern.format("manualgrad", descriptor, lr)
    jaxResFilename = jaxResultsFilenamePattern.format(descriptor)
    scipyHIPSautogradResFilename = scipyResultsFilenamePattern.format("HIPSautograd", descriptor)
    scipyPytorchAutogradResFilename = scipyResultsFilenamePattern.format("pytorchAutograd", descriptor)
    scipyManualgradResFilename = scipyResultsFilenamePattern.format("manualgrad", descriptor)

    matlabRes = np.loadtxt(fname=matlabResFilename, delimiter=",")
    pytorchAutogradRes = np.loadtxt(fname=pytorchAutogradResFilename, delimiter=",")
    pytorchManualGradRes = np.loadtxt(fname=pytorchManualgradResFilename, delimiter=",")
    jaxRes = np.loadtxt(fname=jaxResFilename, delimiter=",")
    scipyPytorchAutogradRes = np.loadtxt(fname=scipyPytorchAutogradResFilename, delimiter=",")
    scipyHIPSautogradRes = np.loadtxt(fname=scipyHIPSautogradResFilename, delimiter=",")
    scipyManualGradRes = np.loadtxt(fname=scipyManualgradResFilename, delimiter=",")

    elapsedTimes = {
        "matlab_manualgrad": matlabRes[:,1],
        "jax_autogradJIT": jaxRes[:,1],
        "scipy_HIPSautograd": scipyHIPSautogradRes[:,1],
        "scipy_pytorchAutograd": scipyPytorchAutogradRes[:,1],
        "scipy_manualgrad": scipyManualGradRes[:,1],
        "pytorch_pyTorchAutograd": pytorchAutogradRes[:,1],
        "pytorch_manualgrad": pytorchManualGradRes[:,1],
        }

    approxError = {
        "matlab_manualgrad": matlabRes[:,0],
        "jax_autogradJIT": jaxRes[:,0],
        "scipy_HIPSautograd": scipyHIPSautogradRes[:,0],
        "scipy_pytorchAutograd": scipyPytorchAutogradRes[:,0],
        "scipy_manualgrad": scipyManualGradRes[:,0],
        "pytorch_autograd": pytorchAutogradRes[:,0],
        "pytorch_manualgrad": pytorchManualGradRes[:,0],
        }

    pio.renderers.default = "browser"

    fig = plotFunctions.getBoxPlots(data=elapsedTimes, ylab="Elapsed Time (sec)", ylim=ylimElapsedTime)
    fig.write_html(figFilenamePattern.format("elapsedTime", descriptor, lr, "html"))
    fig.write_image(figFilenamePattern.format("elapsedTime", descriptor, lr, "png"))
    fig.show()

    fig = plotFunctions.getBoxPlots(data=approxError, ylab="Approximation Error", ylim=ylimApproxError)
    fig.write_html(figFilenamePattern.format("approxError", descriptor, lr, "html"))
    fig.write_image(figFilenamePattern.format("approxError", descriptor, lr, "png"))
    fig.show()

    # pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
