#!/bin/csh

# matlab -nodisplay -nosplash -nodesktop -r "paramsConfigFilename='../data/rosenbrock_difficult.ini';run('doTestMatlabManualgrad.m');exit;"
# ipython --pdb doTestPytorchAutograd.py ../data/rosenbrock_difficult.ini  -- --lr=1.0
# ipython --pdb doTestPytorchManualgrad.py ../data/rosenbrock_difficult.ini  -- --lr=1.0
# ipython --pdb doTestSciPyPyTorchAutograd.py ../data/rosenbrock_difficult.ini
# ipython --pdb doTestSciPyHIPSautograd.py ../data/rosenbrock_difficult.ini
# ipython --pdb doTestJAXSciPy.py ../data/rosenbrock_difficult.ini
# ipython --pdb doTestSciPyManualgrad.py ../data/rosenbrock_difficult.ini
# ipython --pdb doPlots.py rosenbrock_difficult 1.0

# matlab -nodisplay -nosplash -nodesktop -r "paramsConfigFilename='../data/rosenbrock_medium.ini';run('doTestMatlabManualgrad.m');exit;"
# ipython --pdb doTestPytorchAutograd.py ../data/rosenbrock_medium.ini  -- --lr=1.0
# ipython --pdb doTestPytorchManualgrad.py ../data/rosenbrock_medium.ini  -- --lr=1.0
# ipython --pdb doTestSciPyPyTorchAutograd.py ../data/rosenbrock_medium.ini
# ipython --pdb doTestSciPyHIPSautograd.py ../data/rosenbrock_medium.ini
# ipython --pdb doTestJAXSciPy.py ../data/rosenbrock_medium.ini
# ipython --pdb doTestSciPyManualgrad.py ../data/rosenbrock_medium.ini
# ipython --pdb doPlots.py rosenbrock_medium 1.0

# matlab -nodisplay -nosplash -nodesktop -r "paramsConfigFilename='../data/sixHumpCamel.ini';run('doTestMatlabManualgrad.m');exit;"
# ipython --pdb doTestPytorchAutograd.py ../data/sixHumpCamel.ini  -- --lr=1.0
# ipython --pdb doTestPytorchManualgrad.py ../data/sixHumpCamel.ini  -- --lr=1.0
# ipython --pdb doTestSciPyPyTorchAutograd.py ../data/sixHumpCamel.ini
# ipython --pdb doTestSciPyHIPSautograd.py ../data/sixHumpCamel.ini
# ipython --pdb doTestJAXSciPy.py ../data/sixHumpCamel.ini
# ipython --pdb doTestSciPyManualgrad.py ../data/sixHumpCamel.ini
# ipython --pdb doPlots.py sixHumpCamel 1.0  -- --ylimElapsedTime=\[0.0,0.06\]

# matlab -nodisplay -nosplash -nodesktop -r "paramsConfigFilename='../data/zakharov.ini';run('doTestMatlabManualgrad.m');exit;"
# ipython --pdb doTestPytorchAutograd.py ../data/zakharov.ini  -- --lr=1.0
ipython --pdb doTestPytorchManualgrad.py ../data/zakharov.ini  -- --lr=1.0
ipython --pdb doTestSciPyPyTorchAutograd.py ../data/zakharov.ini
ipython --pdb doTestSciPyHIPSautograd.py ../data/zakharov.ini
ipython --pdb doTestJAXSciPy.py ../data/zakharov.ini
ipython --pdb doTestSciPyManualgrad.py ../data/zakharov.ini
ipython --pdb doPlots.py zakharov 1.0 -- --ylimElapsedTime=\[0.0,0.25\]

