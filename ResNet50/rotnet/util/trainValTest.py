import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random


device = 'cuda'

def rotateBatch(batch):
    size = batch.shape[0]
    rotationMult = [i for i in range(4) for j in range(size)]
    random.shuffle(rotationMult)

    for idx, mult in zip(range(size), rotationMult):
        batch[idx,:,:,:] = torch.rot90(batch[idx,:,:,:], mult, [1,2])
    return batch, torch.IntTensor(rotationMult[:size])

def train(optimizer, epoch, model, train_loader, bs, rotate=False):
    model.train()

    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):
        if rotate == True:
            im, labels = rotateBatch(batch_samples['img'])
        else:
            im, labels = batch_samples['img'], batch_samples['label']
        # move data to device
        data, target = im.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)

        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += loss

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)

        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))

    averageLoss = train_loss/len(train_loader.dataset)
    return averageLoss

def val(epoch, model, val_loader, rotate=False):

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
            if rotate == True:
                im, labels =  rotateBatch(batch_samples['img'])
            else:
                im, labels = batch_samples['img'], batch_samples['label']

            data, target = im.to(device), labels.to(device)

            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)

    return targetlist, scorelist, predlist

def test(epoch, model, test_loader, rotate=False):

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
            if rotate == True:
                im, labels =  rotateBatch(batch_samples['img'])
            else:
                im, labels = batch_samples['img'], batch_samples['label']

            data, target = im.to(device), labels.to(device)
            output = model(data)

            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
    return targetlist, scorelist, predlist
