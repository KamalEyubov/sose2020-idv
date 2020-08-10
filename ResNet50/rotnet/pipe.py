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

    for m in methods:
        with open(m) as infile:
            exec(infile.read())
