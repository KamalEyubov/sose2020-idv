This directory includes all the experiments with the DenseNet169 model conducted by Kamal Eyubov.

A shortened naming for the experiments was used:
No SSL Random       = RAND
No SSL Pre-trained  = PRET
Method 1            = XY_1
Method 2            = XY_2
Method 3            = XY_3
Self-Trans          = XY_ST
XY can be either RN (RotNet) or SC (SimCLR).

Each experiment is a single script, stages of which can all be run at once.

For the scripts to be run, there should be no directories in the same folder with them as some directories are being generated during the execution,
which in turn may lead to file name conflicts.
The scripts generate the following directories:
model_backup - contains checkpoints of the model for the fine-tuning stage
ssl_backup - (only in XY_1, XY_2, XY_3) contains checkpoints of the model for the SSL stage
ssl1_backup - (only in XY_ST) contains checkpoints of the model for the SSL stage on LUNA
ssl2_backup - (only in XY_ST) contains checkpoints of the model for the SSL stage on COVID-CT

The scripts also need the datasets in the superdirectory (..) of this directory and the data split file for COVID-CT dataset.

It is assumed that the comments in the scripts will be read in this particular order:
PRED, RAND, RN_1, RN_2, RN_3, RN_ST, SC_1, SC_2, SC_3, SC_ST
Since the scripts are somewhat similar and some line sequences are repeated accross multiple scripts,
the comments explaining those sequences are only written for their first appearances.

Experiments also contain Jupyter Notebook (ipynb) files.
Those files are the original scripts which were run on Google Colaboratory.
Along with them, the Python (py) files were exported.

All scripts are adapted from the file COVID-CT/baseline methods/DenseNet169/DenseNet_predict.py of the repository at https://github.com/UCSD-AI4H/COVID-CT.
