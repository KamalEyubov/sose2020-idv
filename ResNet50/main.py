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

#training process is defined here

alpha = None
## alpha is None if mixup is not used
alpha_name = f'{alpha}'
#device = 'cuda'

if __name__ == '__main__':
    batchsize = 16
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    trainset, valset, testset = getTransformedDataSplit()
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

    # for batch_index, batch_samples in enumerate(train_loader):
    #         data, target = batch_samples['img'], batch_samples['label']
    # skimage.io.imshow(data[0,1,:,:].numpy())
    # plt.savefig('img.png')
    # train

    '''ResNet50 pretrained'''

    import torchvision.models as models
    from resNet import *
    model = resnet50().cuda()

    # checkpoint = torch.load('new_data/save_model/checkpoint.pth.tar')
    # # print(checkpoint.keys())
    # # print(checkpoint['arch'])

    # state_dict = checkpoint['state_dict']
    # for key in list(state_dict.keys()):
    #     if 'module.encoder_q' in key:
    #         print(key[17:])
    #         new_key = key[17:]
    #         state_dict[new_key] = state_dict[key]
    #     del state_dict[key]
    # for key in list(state_dict.keys()):
    #     if  key == 'fc.0.weight':
    #         new_key = 'fc.weight'
    #         state_dict[new_key] = state_dict[key]
    #         del state_dict[key]
    #     if  key == 'fc.0.bias':
    #         new_key = 'fc.bias'
    #         state_dict[new_key] = state_dict[key]
    #         del state_dict[key]
    #     if  key == 'fc.2.weight' or key == 'fc.2.bias':
    #         del state_dict[key]
    # state_dict['fc.weight'] = state_dict['fc.weight'][:1000,:]
    # state_dict['fc.bias'] = state_dict['fc.bias'][:1000]
    # # print(state_dict.keys())

    # # print(state_dict)
    # # pattern = re.compile(
    # #         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    # #     for key in list(state_dict.keys()):
    # #         match = pattern.match(key)
    # #         new_key = match.group(1) + match.group(2) if match else key
    # #         new_key = new_key[7:] if remove_data_parallel else new_key
    # #         new_key = new_key[7:]
    # #         state_dict[new_key] = state_dict[key]
    # #         del state_dict[key]

    # # model.load_state_dict(checkpoint['state_dict'])

    modelname = 'ResNet50'
    # modelname = 'ResNet50_ssl'




    votenum = 1
    import warnings
    warnings.filterwarnings('ignore')

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []
    F1_list = []

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

    scheduler = StepLR(optimizer, step_size=1)

    total_epoch = 1
    for epoch in range(1, total_epoch+1):
        train(optimizer, epoch, model, train_loader, batchsize)

        targetlist, scorelist, predlist = val(epoch, model, val_loader)
        print('target',targetlist)
        print('score',scorelist)
        print('predict',predlist)
        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist



        if epoch % votenum == 0:

            eval = Evaluation(vote_pred, votenum, vote_score, targetlist, predlist, 'train')
            r_list.append(eval.recall)
            p_list.append(eval.precision)
            acc_list.append(eval.getAccuracy())
            AUC_list.append(eval.getAUC())
            F1_list.append(eval.getF1())

            if epoch == total_epoch:
                torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_covid_moco_covid.pt".format(modelname,alpha_name))

                vote_pred = np.zeros(valset.__len__())
                vote_score = np.zeros(valset.__len__())
                print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
        average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                epoch, sum(r_list)/len(r_list), sum(p_list)/len(p_list), sum(F1_list)/len(F1_list), sum(acc_list)/len(acc_list), sum(AUC_list)/len(AUC_list)))

                f = open('model_result/medical_transfer/{}_{}.txt'.format(modelname,alpha_name), 'a+')
                f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
        average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
                epoch, sum(r_list)/len(r_list), sum(p_list)/len(p_list), sum(F1_list)/len(F1_list), sum(acc_list)/len(acc_list), sum(AUC_list)/len(AUC_list)))
                f.close()

    # test
    import warnings
    warnings.filterwarnings('ignore')

    epoch = 1
    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []
    F1_list = []

    vote_pred = np.zeros(testset.__len__())
    vote_score = np.zeros(testset.__len__())


    targetlist, scorelist, predlist = test(epoch, model, test_loader)

    eval = Evaluation(vote_pred, votenum, vote_score, targetlist, predlist, 'test')
    r_list.append(eval.recall)
    p_list.append(eval.precision)
    acc_list.append(eval.getAccuracy())
    AUC_list.append(eval.getAUC())
    F1_list.append(eval.getF1())

    eval.plotEval()
    eval.plotConfusion()
    f = open(f'model_result/medical_transfer/test_{modelname}_{alpha_name}_LUNA_moco_CT_moco.txt', 'a+')
    f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
    average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
    epoch, sum(r_list)/len(r_list), sum(p_list)/len(p_list), sum(F1_list)/len(F1_list), sum(acc_list)/len(acc_list), sum(AUC_list)/len(AUC_list)))
    f.close()
    torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_covid_moco_covid.pt".format(modelname,alpha_name))
