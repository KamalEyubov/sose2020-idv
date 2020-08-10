if __name__ == '__main__':
    methods = ['sslSimClrNoPretrain.py',
               'sslSimClrNoPretrainFinetune.py',
               'sslSimClrWithPretrain.py',
               'sslSimClrWithPretrainFinetune.py',
               'sslSimClrWithPretrainLUNA.py',
               'sslSimClrWithPretrainLUNAFinetune.py',
               'sslSimClrWithPretrainLUNAsslSimClr.py',
               'sslSimClrWithPretrainLUNAsslSimClrFinetune.py']

    for m in methods:
        with open(m) as infile:
            exec(infile.read())
