#import torch
import torchvision
#import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torch.nn.functional as F
#import torch.nn as nn
import torch.optim as optim
#import os
#import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
#import numpy as np
from datetime import datetime
import pandas as pd
import random
from torchvision.datasets import ImageFolder
import re
from torch.utils.data import  DataLoader
#import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, confusion_matrix
from skimage.io import imread, imsave
import skimage
from PIL import ImageFile
#from PIL import Image
#from sklearn.preprocessing import normalize
from evaluation import *
#from covidDataSet import *
from trainValTest import*
from dataloadersGeneration import *

torch.cuda.empty_cache()

if __name__ == '__main__':
    batchsize = 16
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    trainset, valset, testset = getTransformedDataSplit()
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    for batch_index, batch_samples in enumerate(train_loader):
            data, target = batch_samples['img'], batch_samples['label']
    # skimage.io.imshow(data[0,1,:,:].numpy())
    # plt.savefig('img.png')

    # x = torch.arange(8).view(2, 2, 2)
    # plt.imshow((x[0,:,:].numpy()))
    # plt.savefig('x.png')
    # y = torch.rot90(x, 1, [1, 2])
    # plt.imshow((y[0,:,:].numpy()))
    # plt.savefig('y.png')
    #train
    #print(data.shape)
    # print(type(trainset.transform))
    # print(train_loader.dataset.transform.__dict__['transforms'][1].__dict__['degrees'])
    '''ResNet50 pretrained'''

    import torchvision.models as models
    from resNet import *
    path = 'model_backup/medical_transfer/ResNet50_sslRotateWithPretrainLUNAsslRotate_train_covid_moco_covid.pt'
    model = resnet50()
    model.change_cls_number(num_classes=4)
    model.load_state_dict(torch.load(path))
    model.change_cls_number(num_classes=2)
    model.cuda()


    modelname = 'ResNet50'
    alpha = 'sslRotateWithPretrainLUNAsslRotateFinetune'
    # modelname = 'ResNet50_ssl'




    votenum = 10
    import warnings
    warnings.filterwarnings('ignore')

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

    scheduler = StepLR(optimizer, step_size=1)
    eval = Evaluation(vote_pred, votenum, vote_score, alpha)
    # acc = 0
    # cnt = 0
    total_epoch = 50
    for epoch in range(1, total_epoch+1):
        averageLoss = train(optimizer, epoch, model, train_loader, batchsize, rotate=False)

        targetlist, scorelist, predlist = val(epoch, model, val_loader, rotate=False)
        print('target',targetlist)
        print('score',scorelist)
        print('predict',predlist)
        eval.update(predlist, targetlist, scorelist)
        #scheduler.step()

        if epoch % votenum == 0:
            eval.computeStatistics()
            torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_train_covid_moco_covid.pt".format(modelname,alpha))

            # vote_pred = np.zeros(valset.__len__())
            # vote_score = np.zeros(valset.__len__())
            print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
            average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, eval.getRecall(), eval.getPrecision(), eval.getF1(), eval.getAccuracy(), eval.getAUC()))

            f = open('model_result/medical_transfer/train_{}_{}.txt'.format(modelname,alpha), 'a+')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
            average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, eval.getRecall(), eval.getPrecision(), eval.getF1(), eval.getAccuracy(), eval.getAUC()))
            f.close()

    # test
    import warnings
    warnings.filterwarnings('ignore')

    votenum = 1
    vote_pred = np.zeros(testset.__len__())
    vote_score = np.zeros(testset.__len__())

    eval = Evaluation(vote_pred, votenum, vote_score, alpha)

    targetlist, scorelist, predlist = test(total_epoch, model, test_loader, rotate=False)
    eval.update(predlist, targetlist, scorelist)
    eval.computeStatistics()
    #acc = np.sum(targetlist == predlist)/targetlist.shape[0]

    f = open('model_result/medical_transfer/test_{}_{}.txt'.format(modelname,alpha), 'a+')
    f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
    average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
            epoch, eval.getRecall(), eval.getPrecision(), eval.getF1(), eval.getAccuracy(), eval.getAUC()))
    f.close()

    eval.plotEval()
    eval.plotConfusion()
    torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_test_covid_moco_covid.pt".format(modelname,alpha))
