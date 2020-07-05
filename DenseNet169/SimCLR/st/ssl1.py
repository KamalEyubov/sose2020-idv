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
batchsize = 8
train_epochs_start = 0
train_epochs_end = 100


class SimCLR(torch.nn.Module):

    def __init__(self, out_dim=128):
        super(SimCLR, self).__init__()

	# Randomly initialize weights
        densenet = torchvision.models.densenet169(pretrained=True)
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

model = SimCLR()
if device == 'cuda':
    model = model.cuda()
modelname = 'SimCLR'

train_epochs_start = 10
if train_epochs_start > 0:
    model.load_state_dict(
        torch.load('ssl1_backup/{}_{}.pt'.format(
            modelname,
            train_epochs_start
        ))
    )

class NTXentLoss(torch.nn.Module):

    def __init__(self):
        super(NTXentLoss, self).__init__()
        self.masks = dict()
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def get_mask(batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye(2 * batch_size, k=-batch_size)
        l2 = np.eye(2 * batch_size, k=batch_size)
        mask = torch.from_numpy(diag + l1 + l2)
        mask = (1 - mask).type(torch.bool)
        return mask.to(device)

    def forward(self, zi, zj, temperature=0.5):

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


normalize = transforms.Normalize(
    mean=[0.45271412, 0.45271412, 0.45271412],
    std=[0.33165374, 0.33165374, 0.33165374]
)
color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomApply([color_jitter], p=1.0),
    transforms.ToTensor(),
    normalize
])


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class LUNADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
            Args:
                txt_path (string): Path to the txt file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied on a sample.
            File structure:
                - LUNA
                    - img1.png
                    - img2.png
                    - ......
        """
        self.img_list = [os.path.join(root_dir, item) for item in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image1 = self.transform(image)
            image2 = self.transform(image)
        sample = {'img1': image1, 'img2': image2}
        return sample



if __name__ == '__main__':
    dataset = LUNADataset(
        root_dir='../../../LUNA',
        transform=transformer
    )
    print("dataset length:", dataset.__len__())
    data_loader = DataLoader(dataset, batch_size=batchsize, drop_last=False, shuffle=True)

def train(epoch):

    model.train()

    train_loss = 0

    for batch_index, batch_samples in enumerate(data_loader):

        # move data to device
        data1, data2 = batch_samples['img1'].to(device), batch_samples['img2'].to(device)

        output1 = model(data1)
        output2 = model(data2)

        loss = criteria(output1, output2)
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
    f = open('ssl1_result/train_{}.txt'.format(modelname), 'a+')
    f.write('Data set: Average loss: {:.4f}'.format(train_loss / len(data_loader.dataset)))
    f.close()


# training

print('training')

bs = 10
votenum = 5

criteria = NTXentLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(data_loader))

for epoch in range(train_epochs_start + 1, train_epochs_end + 1):
    train(epoch)

    scheduler.step()
    
    if epoch % votenum == 0:
        # checkpoint
        torch.save(model.state_dict(), 'ssl1_backup/{}_{}.pt'.format(modelname, epoch))
