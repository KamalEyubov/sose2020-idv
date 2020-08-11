import os


if __name__ == '__main__':
    methods = ['sslSimClrNoPretrain.py',
               'sslSimClrNoPretrainFinetune.py',
               'sslSimClrWithPretrain.py',
               'sslSimClrWithPretrainFinetune.py',
               'sslSimClrWithPretrainLUNA.py',
               'sslSimClrWithPretrainLUNAFinetune.py',
               'sslSimClrWithPretrainLUNAsslSimClr.py',
               'sslSimClrWithPretrainLUNAsslSimClrFinetune.py']

    if not os.path.exists('model_backup'):
        os.makedirs('model_backup')

    if not os.path.exists('model_result'):
        os.makedirs('model_result')

    for m in methods:
        with open(m) as infile:
            exec(infile.read())
