import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random



device = 'cuda'

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=2048, out_features=128)

    def forward(self, x):
        return self.linear(self.relu(x))


class NTXentLoss(nn.Module):

    def __init__(self):
        super(NTXentLoss, self).__init__()
        self.masks = dict()
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mlp = MLP().cuda()

    @staticmethod
    def get_mask(batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye(2 * batch_size, k=-batch_size)
        l2 = np.eye(2 * batch_size, k=batch_size)
        mask = torch.from_numpy(diag + l1 + l2)
        mask = (1 - mask).type(torch.bool)
        return mask.to(device)

    def forward(self, zi, zj, temperature=0.5):
        zi = self.mlp(zi)
        zj = self.mlp(zj)
        batch_size = zi.shape[-2]
        if batch_size not in self.masks:
            self.masks[batch_size] = NTXentLoss.get_mask(batch_size)

        representations = torch.cat([zi, zj], dim=0)

        similarity_matrix = self.similarity_function(
            representations.unsqueeze(1),
            representations.unsqueeze(0)
        )

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[self.masks[batch_size]].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature

        labels = torch.zeros(2 * batch_size).to(device).long()
        loss = self.criterion(logits, labels)
        return loss

def train(optimizer, epoch, model, train_loader, bs):

    model.train()

    train_loss = 0

    for batch_index, batch_samples in enumerate(train_loader):

        im, labels = batch_samples['img'], batch_samples['label']
        data, target = im.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)

        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += loss

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)

        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))

    averageLoss = train_loss/len(train_loader.dataset)
    return averageLoss

def trainSimClr(optimizer, epoch, model, train_loader, bs):
    mlp = MLP()
    model.train()

    train_loss = 0

    for batch_index, batch_samples in enumerate(train_loader):
        im1, im2 = batch_samples['img1'], batch_samples['img2']
        data1, data2 = im1.to(device), im2.to(device)
        optimizer.zero_grad()
        output1 = model(data1)
        output2 = model(data2)
        criterion = NTXentLoss()
        loss = criterion(output1, output2)

        train_loss += loss
        loss.backward()
        optimizer.step()

        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))

    averageLoss = train_loss/len(train_loader.dataset)
    return averageLoss

def val(epoch, model, val_loader):

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

def test(epoch, model, test_loader):

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
