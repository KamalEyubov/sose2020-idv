import sys
import torch
import torchvision
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
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

    trainset, valset, testset = getTransformedDataSplit()
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    for batch_index, batch_samples in enumerate(train_loader):
            data, target = batch_samples['img'], batch_samples['label']

    model = resnet50()
    model.cuda()
    modelname = 'ResNet50'
    alpha = 'noSslNoPretrain'

    torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_blank_covid.pt".format(modelname,alpha))

    votenum = 10
    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    eval = Evaluation(vote_pred, votenum, vote_score, alpha)
    total_epoch = 50
    for epoch in range(1, total_epoch+1):
        averageLoss = train(optimizer, epoch, model, train_loader, batchsize, rotate=False)
        targetlist, scorelist, predlist = val(epoch, model, val_loader, rotate=False)
        scheduler.step()
        print(optimizer)
        print('target',targetlist)
        print('score',scorelist)
        print('predict',predlist)
        eval.update(predlist, targetlist, scorelist)


        if epoch % votenum == 0:
            eval.computeStatistics()
            torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_train_covid_moco_covid.pt".format(modelname,alpha))

            print('\n epoch: {}, average recall: {:.4f}, average precision: {:.4f},\
            average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, eval.getRecall(), eval.getPrecision(), eval.getF1(), eval.getAccuracy(), eval.getAUC()))

            f = open('model_result/medical_transfer/train_{}_{}.txt'.format(modelname,alpha), 'a+')
            f.write('\n epoch {}, average recall: {:.4f}, average precision: {:.4f},\
            average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                    epoch, eval.getRecall(), eval.getPrecision(), eval.getF1(), eval.getAccuracy(), eval.getAUC()))
            f.close()


    votenum = 1
    vote_pred = np.zeros(testset.__len__())
    vote_score = np.zeros(testset.__len__())

    eval = Evaluation(vote_pred, votenum, vote_score, alpha)

    targetlist, scorelist, predlist = test(total_epoch, model, test_loader, rotate=False)
    eval.update(predlist, targetlist, scorelist)
    eval.computeStatistics()
    acc = np.sum(targetlist == predlist)/targetlist.shape[0]

    f = open('model_result/medical_transfer/test_{}_{}.txt'.format(modelname,alpha), 'a+')
    f.write('\n epoch {}, average recall: {:.4f}, average precision: {:.4f},\
    average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
            epoch, eval.getRecall(), eval.getPrecision(), eval.getF1(), eval.getAccuracy(), eval.getAUC()))
    f.close()

    eval.plotEval()
    eval.plotConfusion()
    torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_test_covid_moco_covid.pt".format(modelname,alpha))