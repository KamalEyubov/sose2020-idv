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

    trainset = getTransformedSimClr()
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)

    model = resnet50(pretrained=True, num_classes=1000)
    model.change_cls_number(num_classes=2048)
    model.cuda()
    modelname = 'ResNet50'
    alpha = 'SslSimClrWithPretrain'

    votenum = 10

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    acc = 0
    cnt = 0
    total_epoch = 50
    for epoch in range(1, total_epoch+1):
        averageLoss = trainSimClr(optimizer, epoch, model, train_loader, batchsize)
        scheduler.step()

        if epoch % votenum == 0:
            torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_train_covid_moco_covid.pt".format(modelname,alpha))

            print('\n The epoch is {}, average loss: {:.4f}'.format(
            epoch,  averageLoss))

            f = open('model_result/medical_transfer/train_{}_{}.txt'.format(modelname,alpha), 'a+')
            f.write('\n The epoch is {}, average loss: {:.4f}'.format(epoch, averageLoss))
            f.close()
