import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import pandas as pd
import random
from torchvision.datasets import ImageFolder
import re
from torch.utils.data import  DataLoader
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, confusion_matrix
from skimage.io import imread, imsave
import skimage
from PIL import ImageFile
from util.dataloadersGeneration import *
from util.trainValTest import *
from util.evaluation import *
import torchvision.models as models
from resNet import *
import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()

if __name__ == '__main__':
    batchsize = 16

    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    rng = np.random.RandomState(42)

    trainset, valset, testset = getTransformedDataSplit()
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    model = resnet50()
    model.change_cls_number(num_classes=4)
    model.cuda()
    modelname = 'ResNet50'
    alpha = 'sslRotateNoPretrain'

    votenum = 5
    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    optimizer = optim.Adam(model.parameters())

    #print header of statistics file
    f = open('model_result/train_{}_{}.txt'.format(modelname,alpha), 'a+')
    f.write('epoch, average accuracy, loss\n')
    f.close()

    acc = 0
    cnt = 0
    total_epoch = 50
    for epoch in range(1, total_epoch+1):
        averageLoss = train(optimizer, epoch, model, train_loader, batchsize, rotate=True)
        targetlist, scorelist, predlist = val(epoch, model, val_loader, rotate=True)
        print('target',targetlist)
        print('score',scorelist)
        print('predict',predlist)

        if epoch % votenum == 0:
            acc += np.sum(targetlist == predlist)/targetlist.shape[0]
            cnt += 1

            vote_pred = np.zeros(valset.__len__())
            vote_score = np.zeros(valset.__len__())
            print('\n The epoch is {}, average accuracy: {:.4f}, average loss: {:.4f}'.format(
            epoch, acc/cnt,  averageLoss))

            f = open('model_result/train_{}_{}.txt'.format(modelname,alpha), 'a+')
            f.write('{}, {:.4f}, {:.4f}\n'.format(epoch, acc/cnt, averageLoss))
            f.close()
    torch.save(model.state_dict(), "model_backup/{}_{}_train_covid.pt".format(modelname,alpha))

    vote_pred = np.zeros(testset.__len__())
    vote_score = np.zeros(testset.__len__())
    targetlist, scorelist, predlist = test(total_epoch, model, test_loader, rotate=True)
    acc = np.sum(targetlist == predlist)/targetlist.shape[0]

    f = open('model_result/test_{}_{}.txt'.format(modelname,alpha), 'a+')
    f.write('\n The epoch is {}, average accuracy: {:.4f}'.format(epoch, acc))
    f.close()
    torch.save(model.state_dict(), "model_backup/{}_{}_test_covid.pt".format(modelname,alpha))
