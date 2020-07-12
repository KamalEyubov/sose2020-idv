import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet169 as DenseNet169
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


torch.cuda.empty_cache()

device = 'cpu'
batchsize = 16
train_epochs_start = 0
train_epochs_end = 50
ssl_epochs = 50


class RotNet(torch.nn.Module):

    def __init__(self, out_dim=4):
        super(RotNet, self).__init__()

        densenet = torchvision.models.densenet169(pretrained=False)
        num_features = densenet.classifier.in_features

        self.features = torch.nn.Sequential(*list(densenet.children())[: -1])

        self.l1 = torch.nn.Linear(num_features, num_features)
        self.l2 = torch.nn.Linear(num_features, out_dim)

    def forward(self, x):
        out = self.features(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        out = self.l1(out)
        out = F.relu(out)
        out = self.l2(out)
        return out

class FineTune(torch.nn.Module):

    def __init__(self, ssl_model, out_dim=2):
        super(FineTune, self).__init__()

        num_features = ssl_model.l1.in_features
        self.features = ssl_model.features
        self.classifier = torch.nn.Linear(num_features, out_dim)

    def forward(self, x):
        out = self.features(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        out = self.classifier(out)
        return out

model = RotNet()
modelname = 'RotNet'

if train_epochs_start > 0:
    model = FineTune(model)
    modelname = 'FineTune'
    model.load_state_dict(
        torch.load('model_backup/{}_{}.pt'.format(
            modelname,
            train_epochs_start
        ))
    )
else:
    model.load_state_dict(
        torch.load('ssl_backup/{}_{}.pt'.format(
            modelname,
            ssl_epochs
        ))
    )
    model = FineTune(model)
    modelname = 'FineTune'

if device == 'cuda':
    model = model.cuda()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


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
                transform (callable, optional): Optional transform to be applied on a sample.
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
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
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
        sample = {'img': image, 'label': int(self.img_list[idx][1])}
        return sample



if __name__ == '__main__':
    trainset = CovidCTDataset(
        root_dir='../../../Images',
        txt_COVID='../../../Data-split/COVID/trainCT_COVID.txt',
        txt_NonCOVID='../../../Data-split/NonCOVID/trainCT_NonCOVID.txt',
        transform=train_transformer
    )
    valset = CovidCTDataset(
        root_dir='../../../Images',
        txt_COVID='../../../Data-split/COVID/valCT_COVID.txt',
        txt_NonCOVID='../../../Data-split/NonCOVID/valCT_NonCOVID.txt',
        transform=val_transformer
    )
    testset = CovidCTDataset(
        root_dir='../../../Images',
        txt_COVID='../../../Data-split/COVID/testCT_COVID.txt',
        txt_NonCOVID='../../../Data-split/NonCOVID/testCT_NonCOVID.txt',
        transform=val_transformer
    )
    print("trainset length:", trainset.__len__())
    print("valset length:", valset.__len__())
    print("testset length:", testset.__len__())

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)


def train(epoch):
        
    model.train()
        
    train_loss = 0
    train_correct = 0
        
    for batch_index, batch_samples in enumerate(train_loader):
                
        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        output = model(data)
        loss = criteria(output, target.long())
        train_loss += loss.item()
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        train_correct += (pred == target).sum().item()
        
        # Display progress
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch,
                batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader),
                train_loss / bs
            ))
            train_loss = 0
        
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset),
        train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)
    ))
    f.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(train_loader.dataset),
        train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)
    ))


def predict(mode):
    '''
        mode:
            either 'val' or 'train'
    '''
    loader = val_loader if mode == 'val' else test_loader
        
    model.eval()

    predlist=[]
    scorelist=[]
    targetlist=[]
        
    # Don't update model
    with torch.no_grad():
        # Predict
        for batch_index, batch_samples in enumerate(loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output = model(data)
                        
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1]) # percentages for positive
            targetlist=np.append(targetlist, targetcpu)
                                  
    return targetlist, scorelist, predlist


# train

print('training')

bs = 10
votenum = 10

f = open('model_result/train_{}.txt'.format(modelname), 'a+')

vote_pred = np.zeros(valset.__len__())
vote_score = np.zeros(valset.__len__())

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(train_epochs_start + 1, train_epochs_end + 1):
    train(epoch)

    targetlist, scorelist, predlist = predict('val')
    vote_pred = vote_pred + predlist
    vote_score = vote_score + scorelist

    scheduler.step()

    if epoch % votenum == 0:
        # checkpoint
        torch.save(model.state_dict(), "model_backup/{}_{}.pt".format(modelname, epoch))
                
        # major vote
        vote_pred[vote_pred <=  (votenum / 2)] = 0
        vote_pred[vote_pred >   (votenum / 2)] = 1
        vote_score = vote_score / votenum

        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
                
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP + FP)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        AUC = roc_auc_score(targetlist, vote_score)

        print('The epoch is {}, average recall: {:.4f}, average precision: {:.4f}, average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n\n'.format(
            epoch, r, p, F1, acc, AUC
        ))
        f.write('The epoch is {}, average recall: {:.4f}, average precision: {:.4f}, average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n\n'.format(
            epoch, r, p, F1, acc, AUC
        ))
                
        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())

f.close()


# test

print('testing')

bs = 10

f = open('model_result/test_{}.txt'.format(modelname), 'a+')
        
targetlist, scorelist, predlist = predict('test')
        
TP = ((predlist == 1) & (targetlist == 1)).sum()
TN = ((predlist == 0) & (targetlist == 0)).sum()
FN = ((predlist == 0) & (targetlist == 1)).sum()
FP = ((predlist == 1) & (targetlist == 0)).sum()

print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
print('TP+FP',TP+FP)
p = TP / (TP + FP)
r = TP / (TP + FN)
F1 = 2 * r * p / (r + p)
acc = (TP + TN) / (TP + TN + FP + FN)
AUC = roc_auc_score(targetlist, scorelist)

print('Testing, average recall: {:.4f}, average precision: {:.4f}, average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
    r, p, F1, acc, AUC
))
f.write('Testing, average recall: {:.4f}, average precision: {:.4f}, average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
    r, p, F1, acc, AUC
))

f.close()
