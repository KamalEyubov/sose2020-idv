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
batchsize = 4
train_epochs_start = 0
train_epochs_end = 50


class RotNet(torch.nn.Module):

    def __init__(self, out_dim=4):
        super(RotNet, self).__init__()

	# Randomly initialize weights
        densenet = torchvision.models.densenet169(pretrained=False)
        num_features = densenet.classifier.in_features

        self.features = torch.nn.Sequential(*list(densenet.children())[: -1])

        # projection MLP
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

model = RotNet()
if device == 'cuda':
    model = model.cuda()
modelname = 'RotNet'

if train_epochs_start > 0:
    model.load_state_dict(
        torch.load('ssl_backup/{}_{}.pt'.format(
            modelname,
            train_epochs_start
        ))
    )


normalize = transforms.Normalize(
    mean=[0.45271412, 0.45271412, 0.45271412],
    std=[0.33165374, 0.33165374, 0.33165374]
)
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
def rotate(tensor, angle):
    if angle == 0:
        return tensor
    elif angle == 90:
        return tensor.transpose(-1, -2).flip((-2))
    elif angle == 180:
        return tensor.flip((-1, -2))
    elif angle == 270:
        return tensor.flip((-2)).transpose(-1, -2)
    else:
        return None


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=transformer):
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
            cls_list = [os.path.join(self.root_dir, self.classes[c], item) for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)
        image0 = rotate(image, 0)
        image1 = rotate(image, 90)
        image2 = rotate(image, 180)
        image3 = rotate(image, 270)

        sample = {
            'img0': image0, 'label0': 0,
            'img1': image1, 'label1': 1,
            'img2': image2, 'label2': 2,
            'img3': image3, 'label3': 3
        }
        return sample



if __name__ == '__main__':
    dataset = CovidCTDataset(
        root_dir='../../../Images',
        txt_COVID='../../../Data-split/COVID/trainCT_COVID.txt',
        txt_NonCOVID='../../../Data-split/NonCOVID/trainCT_NonCOVID.txt',
        transform=transformer
    )
    print("dataset length:", dataset.__len__())
    data_loader = DataLoader(dataset, batch_size=batchsize, drop_last=False, shuffle=True)

def train(epoch):

    model.train()

    train_loss = 0

    for batch_index, batch_samples in enumerate(data_loader):

        # move data to device
        data0, label0 = batch_samples['img0'].to(device), batch_samples['label0'].to(device)
        data1, label1 = batch_samples['img1'].to(device), batch_samples['label1'].to(device)
        data2, label2 = batch_samples['img2'].to(device), batch_samples['label2'].to(device)
        data3, label3 = batch_samples['img3'].to(device), batch_samples['label3'].to(device)

        output0 = model(data0)
        output1 = model(data1)
        output2 = model(data2)
        output3 = model(data3)

        loss0 = criteria(output0, label0.long())
        loss1 = criteria(output1, label1.long())
        loss2 = criteria(output2, label2.long())
        loss3 = criteria(output3, label3.long())
        loss = (loss0 + loss1 + loss2 + loss3) / 4
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Display progress
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch,
                batch_index, len(data_loader),
                100.0 * batch_index / len(data_loader)
            ))

    print('Data set: Average loss: {:.4f}'.format(train_loss / len(data_loader.dataset)))
    f = open('ssl_result/train_{}.txt'.format(modelname), 'a+')
    f.write('Data set: Average loss: {:.4f}'.format(train_loss / len(data_loader.dataset)))
    f.close()


# training

print('training')

bs = 10
votenum = 10

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_loader))

for epoch in range(train_epochs_start + 1, train_epochs_end + 1):
    train(epoch)

    scheduler.step()
    
    if epoch % votenum == 0:
        # checkpoint
        torch.save(model.state_dict(), 'ssl_backup/{}_{}.pt'.format(modelname, epoch))
