import os


if __name__ == '__main__':
    methods = ['noSslNoPretrain.py',
               'noSslWithPretrain.py',
               'sslRotateNoPretrain.py',
               'sslRotateNoPretrainFinetune.py',
               'sslRotateWithPretrain.py',
               'sslRotateWithPretrainFinetune.py',
               'sslRotateWithPretrainLUNA.py',
               'sslRotateWithPretrainLUNAFinetune.py',
               'sslRotateWithPretrainLUNAsslRotate.py',
               'sslRotateWithPretrainLUNAsslRotateFinetune.py']

    if not os.path.exists('model_backup'):
        os.makedirs('model_backup')

    if not os.path.exists('model_result'):
        os.makedirs('model_result')

    for m in methods:
        with open(m) as infile:
            exec(infile.read())
