import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from datetime import datetime
import pandas as pd
import random
from torchvision.datasets import ImageFolder
import re
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, confusion_matrix
from skimage.io import imread, imsave
import skimage
from PIL import ImageFile
from PIL import Image
from sklearn.preprocessing import normalize
from evaluation import *

torch.cuda.empty_cache()


batchsize=4
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample

#training process is defined here

alpha = None
## alpha is None if mixup is not used
alpha_name = f'{alpha}'
device = 'cuda'

def train(optimizer, epoch):

    model.train()

    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        ## adjust data to meet the input dimension of model
#         data = data[:, 0, :, :]
#         data = data[:, None, :, :]

        #mixup
#         data, targets_a, targets_b, lam = mixup_data(data, target, alpha, use_cuda=True)


        optimizer.zero_grad()
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())

        #mixup loss
#         loss = mixup_criterion(criteria, output, targets_a, targets_b, lam)

        train_loss += criteria(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))

#     print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
#         100.0 * train_correct / len(train_loader.dataset)))
#     f = open('model_result/{}.txt'.format(modelname), 'a+')
#     f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
#         100.0 * train_correct / len(train_loader.dataset)))
#     f.write('\n')
#     f.close()

def val(epoch):

    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0


    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

#             data = data[:, 0, :, :]
#             data = data[:, None, :, :]
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
#             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()

#             print(output[:,1].cpu().numpy())
#             print((output[:,1]+output[:,0]).cpu().numpy())
#             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)

    return targetlist, scorelist, predlist

    # Write to tensorboard
#     writer.add_scalar('Test Accuracy', 100.0 * correct / len(test_loader.dataset), epoch)

def test(epoch):

    model.eval()
    test_loss = 0
    correct = 0
    results = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0


    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []

        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
#             data = data[:, 0, :, :]
#             data = data[:, None, :, :]
#             print(target)
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
#             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()
#             TP += ((pred == 1) & (target.long()[:, 2].view_as(pred).data == 1)).cpu().sum()
#             TN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
# #             # FN    predict 0 label 1
#             FN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 1)).cpu().sum()
# #             # FP    predict 1 label 0
#             FP += ((pred == 1) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
#             print(TP,TN,FN,FP)


#             print(output[:,1].cpu().numpy())
#             print((output[:,1]+output[:,0]).cpu().numpy())
#             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
    return targetlist, scorelist, predlist

    # Write to tensorboard
#     writer.add_scalar('Test Accuracy', 100.0 * correct / len(test_loader.dataset), epoch)


if __name__ == '__main__':
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ########## Mean and std are calculated from the train dataset
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                         std=[0.33165374, 0.33165374, 0.33165374])
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(90),
        # random brightness and random contrast
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    trainset = CovidCTDataset(root_dir='data',
                              txt_COVID='data/covidSplit/trainCT_COVID.txt',
                              txt_NonCOVID='data/nonCovidSplit/trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='data',
                              txt_COVID='data/covidSplit/valCT_COVID.txt',
                              txt_NonCOVID='data/nonCovidSplit/valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir='data',
                              txt_COVID='data/covidSplit/testCT_COVID.txt',
                              txt_NonCOVID='data/nonCovidSplit/testCT_NonCOVID.txt',
                              transform= val_transformer)
    print(trainset.__len__())
    print(valset.__len__())
    print(testset.__len__())

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




    bs =batchsize
    votenum = 10
    import warnings
    warnings.filterwarnings('ignore')

    r_list = []
    p_list = []
    acc_list = []
    AUC_list = []
    # TP = 0
    # TN = 0
    # FN = 0
    # FP = 0
    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

    scheduler = StepLR(optimizer, step_size=1)

    total_epoch = 1
    for epoch in range(1, total_epoch+1):
        train(optimizer, epoch)

        targetlist, scorelist, predlist = val(epoch)
        print('target',targetlist)
        print('score',scorelist)
        print('predict',predlist)
        vote_pred = vote_pred + predlist
        vote_score = vote_score + scorelist


        eval = Evaluation(vote_pred, votenum, vote_score, targetlist, predlist, valset, 'train')
        eval.plotEval()
        eval.plotConfusion()


        if epoch % votenum == 0:

            # major vote
            vote_pred[vote_pred <= (votenum/2)] = 0
            vote_pred[vote_pred > (votenum/2)] = 1
            vote_score = vote_score/votenum

            print('vote_pred', vote_pred)
            print('targetlist', targetlist)
            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()


            print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
            print('TP+FP',TP+FP)
            p = TP / (TP + FP)
            print('precision',p)
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            print('recall',r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1',F1)
            print('acc',acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUCp', roc_auc_score(targetlist, vote_pred))
            print('AUC', AUC)



    #         if epoch == total_epoch:
    #         torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_covid_moco_covid.pt".format(modelname,alpha_name))
    #
    #         vote_pred = np.zeros(valset.__len__())
    #         vote_score = np.zeros(valset.__len__())
    #         print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
    # average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
    #         epoch, r, p, F1, acc, AUC))
    #
    #         f = open('model_result/medical_transfer/{}_{}.txt'.format(modelname,alpha_name), 'a+')
    #         f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
    # average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
    #         epoch, r, p, F1, acc, AUC))
    #         f.close()

    # test
    bs = 10
    import warnings
    warnings.filterwarnings('ignore')

    epoch = 1
    # r_list = []
    # p_list = []
    # acc_list = []
    # AUC_list = []
    # TP = 0
    # TN = 0
    # FN = 0
    # FP = 0
    vote_pred = np.zeros(testset.__len__())
    vote_score = np.zeros(testset.__len__())


    targetlist, scorelist, predlist = test(epoch)

    eval = Evaluation(vote_pred, votenum, vote_score, targetlist, predlist, valset, 'test')
    eval.plotEval()
    eval.plotConfusion()
    # print('target',targetlist)
    # print('score',scorelist)
    # print('predict',predlist)
    # vote_pred = vote_pred + predlist
    # vote_score = vote_score + scorelist
    #
    # TP = ((predlist == 1) & (targetlist == 1)).sum()
    #
    # TN = ((predlist == 0) & (targetlist == 0)).sum()
    # FN = ((predlist == 0) & (targetlist == 1)).sum()
    # FP = ((predlist == 1) & (targetlist == 0)).sum()
    #
    # print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    # print('TP+FP',TP+FP)
    # p = TP / (TP + FP)
    # print('precision',p)
    # p = TP / (TP + FP)
    # r = TP / (TP + FN)
    # print('recall',r)
    # F1 = 2 * r * p / (r + p)
    # acc = (TP + TN) / (TP + TN + FP + FN)
    # print('F1',F1)
    # print('acc',acc)
    # AUC = roc_auc_score(targetlist, vote_score)
    # print('AUC', AUC)

    # f = open(f'model_result/medical_transfer/test_{modelname}_{alpha_name}_LUNA_moco_CT_moco.txt', 'a+')
    # f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},\
    # average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
    # epoch, r, p, F1, acc, AUC))
    # f.close()
    # torch.save(model.state_dict(), "model_backup/medical_transfer/{}_{}_covid_moco_covid.pt".format(modelname,alpha_name))
