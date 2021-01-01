#!/bin/csh

set oldPattern = manualGrad
set newPattern = Manualgrad

foreach oldName ($*)
    set newName = `echo $oldName | sed "s/$oldPattern/$newPattern/"`
    mv $oldName $newName
end

