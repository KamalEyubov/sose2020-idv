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
from util.resNet import *
import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()

if __name__ == '__main__':
    batchsize = 16

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    rng = np.random.RandomState(42)

    trainset = getTransformedLUNA()
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)

    model = resnet50()
    model.change_cls_number(num_classes=2048)
    model.cuda()
    modelname = 'ResNet50'
    alpha = 'sslSimClrWithPretrainLuna'

    votenum = 5

    optimizer = optim.Adam(model.parameters())

    #print header of statistics file
    f = open('model_result/train_{}_{}.txt'.format(modelname,alpha), 'a+')
    f.write('epoch, loss\n')
    f.close()

    total_epoch = 50
    for epoch in range(1, total_epoch+1):
        averageLoss = trainSimClr(optimizer, epoch, model, train_loader, batchsize)

        if epoch % votenum == 0:

            print('\n The epoch is {}, average loss: {:.4f}'.format(
            epoch,  averageLoss))

            f = open('model_result/train_{}_{}.txt'.format(modelname,alpha), 'a+')
            f.write('{}, {:.4f}\n'.format(epoch, averageLoss))
            f.close()
    torch.save(model.state_dict(), "model_backup/{}_{}_train_covid.pt".format(modelname,alpha))
