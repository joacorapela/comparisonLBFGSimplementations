addpath(genpath('../src'))
addpath(genpath('../../matlabCode/src/util/minFunc'))
addpath(genpath('~/dev/research/programs/src/matlab/iniconfig'))

resultsFilenamePattern='../results/matlab_manualgrad_%s.csv';

if ~exist('paramsConfigFilename','var')
    error('Variable paramsConfigFilename should be defined before calling this script');
end

[filepath,descriptor,ext] = fileparts(paramsConfigFilename);

resultsFilename = sprintf(resultsFilenamePattern, descriptor);

ini = IniConfig();
[~] = ini.ReadFile(paramsConfigFilename);
minima = ini.GetValues('data_params', 'minima');
minima = regexprep(minima, '[', '');
minima = regexprep(minima, ']', '');
minima = str2num(minima);
x0 = ini.GetValues('init_params', 'x0');
x0 = regexprep(x0, '[', '');
x0 = regexprep(x0, ']', '');
x0 = str2num(x0);
x0 = x0';
minima = reshape(minima, length(x0), []);
maxIter = ini.GetValues('optim_params', 'maxIter');
toleranceGrad = str2double(ini.GetValues('optim_params', 'toleranceGrad'));
toleranceChange = str2double(ini.GetValues('optim_params', 'toleranceChange'));
nRepeats = ini.GetValues('test_params', 'nRepeats');

optimopts = optimset('Gradobj','on','display', 'none');
optimopts.MaxIter = maxIter;
optimopts.TolFun = toleranceGrad;
optimopts.TolX = toleranceChange;

if contains(descriptor, 'rosenbrock')
    func = @rosenbrockWithGrad;
elseif strcmp(descriptor, 'sixHumpCamel')
    func = @sixHumpCamelWithGrad;
elseif strcmp(descriptor, 'zakharov')
    func = @zakharovWithGrad;
else
    error(sprintf('Invalid descriptor: %s', descriptor));
end

results = zeros(nRepeats, 2);
for i=1:nRepeats
    startTime = tic;
    [x, fValue, exitfag, output] = minFunc(func, x0, optimopts);
    elapsedTime = toc(startTime);
    results(i,1) = minL2NormToMinima(x, minima);
    results(i,2) = elapsedTime;
end
dlmwrite(resultsFilename, results, 'delimiter', ',', 'precision', 9);

function minL2Norm = minL2NormToMinima(x, minima)
    minL2Norm = norm(x-minima(:,1), 2);
    for i=2:size(minima, 2)
        l2Norm = norm(x-minima(:,i), 2);
        if(l2Norm<minL2Norm)
            minL2Norm = l2Norm;
        end
    end
end

