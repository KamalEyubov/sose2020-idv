# -*- coding: utf-8 -*-
"""DN169_RN_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HS3Y4edEJ6y-Uw-oqnLhO-vzl9V1mHU-

# SSL on LUNA
"""

import os

os.mkdir('ssl_backup')
os.mkdir('ssl_result')

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torchvision.models import densenet169 as DenseNet169
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
rng = np.random.RandomState(42)

torch.cuda.empty_cache()

device = 'cuda'
batchsize = 4
train_epochs_start = 0
train_epochs_end = 50

class RotNet(nn.Module):

    def __init__(self, out_dim=4):
        super(RotNet, self).__init__()

        densenet = DenseNet169(pretrained=True)
        self.features = densenet.features
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(
            densenet.classifier.in_features,
            out_features=out_dim,
            bias=True
        )

    def forward(self, x):
        out = self.features(x)
        out = self.relu(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
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

stages = ['ssl', 'fine-tuning']
stage_id = 0

def plot_change(model1, model2):
    global stage_id

    diff = torch.nn.MSELoss()
    norm_w, norm_wx, norm_b, norm_bx, conv_w, conv_wx = [], [], [], [], [], []
    i = 0
    for (name, parameter1), parameter2 in  zip(model1.named_parameters(), model2.parameters()):
        name = name.split(".")
        if name[-2][:4] == "conv":
            if name[-1] == "weight":
                conv_w.append(diff(parameter1, parameter2).item())
                conv_wx.append(i)
        elif name[-2][:4] == "norm":
            if name[-1] == "weight":
                norm_w.append(diff(parameter1, parameter2).item())
                norm_wx.append(i)
            elif name[-1] == "bias":
                norm_b.append(diff(parameter1, parameter2).item())
                norm_bx.append(i)
        i = i + 1

    plt.bar(norm_wx, norm_w, label="gamma")
    plt.bar(norm_bx, norm_b, label="beta")
    plt.legend()
    plt.title('Change of parameters at BN layers')
    plt.savefig(stages[stage_id] + '_bn_change.png')
    plt.show()

    plt.bar(conv_wx, conv_w)
    plt.title('Change of weights at CONV layers')
    plt.savefig(stages[stage_id] + '_conv_change.png')
    plt.show()

    stage_id = stage_id + 1

model_old = RotNet()
model_old.load_state_dict(model.state_dict())

# Mean and standart deviation were calculated for LUNA.
normalize = transforms.Normalize(
    mean=[0.3226, 0.3226, 0.3226],
    std=[0.3404, 0.3404, 0.3404]
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

detransformer = torchvision.transforms.Compose([
    lambda image_tensor : image_tensor * torch.tensor([0.3226, 0.3226, 0.3226])[:, None, None],
    lambda image_tensor : image_tensor + torch.tensor([0.3404, 0.3404, 0.3404])[:, None, None],
    torchvision.transforms.ToPILImage()
])

def gradcam(model, input_tensor, detransform, label, axis):

	# Activations are saved.
	activations = dict() # one-key dictionary. Doesn't work otherwise.
	def get_activations(name):
		def hook(module, input, output):
			activations[name] = output.detach()
		return hook

	# Gradients are saved.
	grads = dict() # same as for activations
	def get_grads(name):
		def hook(module, grad_input, grad_output):
			grads[name] = grad_output[0].detach() # is a tuple
		return hook

	input_batch = input_tensor.unsqueeze(0) # a batch of one

	frw_handle = model.relu.register_forward_hook(get_activations("features"))
	bck_handle = model.relu.register_backward_hook(get_grads("features"))

	model.eval()
	model.zero_grad()
	output_batch = model(input_batch)
	output_batch[0][label].backward()

	frw_handle.remove()
	bck_handle.remove()

	activations = activations["features"]
	grads = grads["features"]

	grads = torch.mean(grads, dim=(-2, -1), keepdims=True) # gradients are pooled
	grad_cam = activations * grads
	grad_cam = torch.sum(grad_cam, dim=-3) # weighted sum
	grad_cam = torch.nn.functional.relu(grad_cam) # relu'd weighted sum
	grad_cam = grad_cam / grad_cam.max() # normalization
	grad_cam = grad_cam[0] # squeezing the batch of one

	axis.imshow(detransform(input_tensor).resize((224, 224)), alpha=1.0)
	# Grad-CAM is resized to 224x224 via bilinear interpolation and mapped to 'JET' colormap to be superimposed on the original image.
	axis.imshow(torchvision.transforms.ToPILImage()(grad_cam).resize((224, 224), resample=Image.BILINEAR), cmap='jet', alpha=0.5)
	axis.set_xticks([])
	axis.set_yticks([])

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
	# All file names are gathered into a list. Labels (positive or negative) are not needed.
        self.img_list = [os.path.join(root_dir, item) for item in os.listdir(root_dir)]
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

dataset = LUNADataset(
    root_dir='../../LUNA',
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
        loss = (loss0 + loss1 + loss2 + loss3) / 4.
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Display progress
        if (batch_index + 1) % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch,
                batch_index, len(data_loader),
                100.0 * batch_index / len(data_loader)
            ))

    avg_loss = train_loss / len(data_loader.dataset)
    print('Data set: Average loss: {:.4f}'.format(avg_loss))
    f = open('ssl_result/train_{}.txt'.format(modelname), 'a+')
    f.write('Data set: Average loss: {:.4f}'.format(avg_loss))
    f.close()
    avg_losses.append(avg_loss)

bs = 10
T_checkpoint = 10
avg_losses = []

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(train_epochs_start + 1, train_epochs_end + 1):
    train(epoch)
    
    if epoch % T_checkpoint == 0:
        # checkpoint
        torch.save(model.state_dict(), 'ssl_backup/{}_{}.pt'.format(modelname, epoch))

plt.plot(range(train_epochs_start, train_epochs_end), avg_losses)
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.savefig('ssl_loss.png')
plt.show()

plot_change(model.cpu(), model_old)

model = model.cpu()
fig, axes = plt.subplots(3, 4)
for i in range(3):
    samples = dataset[i * 200] # so that the images are different
    image0, label0 = samples['img0'], samples['label0']
    image1, label1 = samples['img1'], samples['label1']
    image2, label2 = samples['img2'], samples['label2']
    image3, label3 = samples['img3'], samples['label3']
    gradcam(model, image0, detransformer, label0, axes[i][0])
    gradcam(model, image1, detransformer, label1, axes[i][1])
    gradcam(model, image2, detransformer, label2, axes[i][2])
    gradcam(model, image3, detransformer, label3, axes[i][3])

fig.savefig('ssl_gradcam.png')
fig.show()

torch.save(model.state_dict(), 'ssl.pt')

"""# Fine-tuning the model on the final task"""

os.mkdir('model_backup')
os.mkdir('model_result')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
rng = np.random.RandomState(42)

from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

torch.cuda.empty_cache()


batchsize = 16
train_epochs_start = 0
train_epochs_end = 50
ssl_epochs = 50

class FineTune(torch.nn.Module):
    def __init__(self, ssl_model, out_dim=2):
        super(FineTune, self).__init__()

        self.features = ssl_model.features
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(
            ssl_model.classifier.in_features,
            out_features=out_dim,
            bias=True
        )

    def forward(self, x):
        out = self.features(x)
        out = self.relu(out)
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

model_old = FineTune(RotNet())
model_old.load_state_dict(model.state_dict())

normalize = transforms.Normalize(
    mean=[0.45271412, 0.45271412, 0.45271412],
    std=[0.33165374, 0.33165374, 0.33165374]
)
train_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

detransformer = torchvision.transforms.Compose([
    lambda image_tensor : image_tensor * torch.tensor([0.33165374, 0.33165374, 0.33165374])[:, None, None],
    lambda image_tensor : image_tensor + torch.tensor([0.45271412, 0.45271412, 0.45271412])[:, None, None],
    torchvision.transforms.ToPILImage()
])

# This definition overrides the previous one
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

trainset = CovidCTDataset(
    root_dir='../../Images',
    txt_COVID='../../Data-split/COVID/trainCT_COVID.txt',
    txt_NonCOVID='../../Data-split/NonCOVID/trainCT_NonCOVID.txt',
    transform=train_transformer
)
valset = CovidCTDataset(
    root_dir='../../Images',
    txt_COVID='../../Data-split/COVID/valCT_COVID.txt',
    txt_NonCOVID='../../Data-split/NonCOVID/valCT_NonCOVID.txt',
    transform=val_transformer
)
testset = CovidCTDataset(
    root_dir='../../Images',
    txt_COVID='../../Data-split/COVID/testCT_COVID.txt',
    txt_NonCOVID='../../Data-split/NonCOVID/testCT_NonCOVID.txt',
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
        if (batch_index + 1) % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch,
                batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader),
                train_loss / bs
            ))
            train_loss = 0
        
    avg_loss = train_loss / len(train_loader.dataset)
    accuracy = 100.0 * train_correct / len(train_loader.dataset)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss,
        train_correct, len(train_loader.dataset),
        accuracy
    ))
    f = open('model_result/train_{}.txt'.format(modelname), 'a+')
    f.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss,
        train_correct, len(train_loader.dataset),
        accuracy
    ))
    f.close()
    avg_losses.append(avg_loss)

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

bs = 10
votenum = 5
epochs, rs, ps, F1s, accs, AUCs = [], [], [], [], [], []
avg_losses = []

vote_pred = np.zeros(valset.__len__())
vote_score = np.zeros(valset.__len__())

criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(train_epochs_start + 1, train_epochs_end + 1):
    train(epoch)

    targetlist, scorelist, predlist = predict('val')
    vote_pred = vote_pred + predlist
    vote_score = vote_score + scorelist

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
        f = open('model_result/train_{}.txt'.format(modelname), 'a+')
        f.write('The epoch is {}, average recall: {:.4f}, average precision: {:.4f}, average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n\n'.format(
            epoch, r, p, F1, acc, AUC
        ))
        f.close()

        epochs.append(epoch)
        rs.append(r)
        ps.append(p)
        F1s.append(F1)
        accs.append(acc)
        AUCs.append(AUC)
                
        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())

plt.plot(epochs, rs, label='Recall')
plt.plot(epochs, ps, label='Precision')
plt.plot(epochs, F1s, label='F1')
plt.plot(epochs, accs, label='Accuracy')
plt.plot(epochs, AUCs, label='AUC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Training result')
plt.xticks(epochs)
plt.savefig('validation_performance.png')
plt.show()

plt.plot(range(train_epochs_start, train_epochs_end), avg_losses)
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.title('Training loss during fine-tuning')
plt.savefig('fine-tuning_loss.png')
plt.show()

plot_change(model.cpu(), model_old)
if device == 'cuda':
    model = model.cuda()

bs = 10

        
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
f = open('model_result/test_{}.txt'.format(modelname), 'a+')
f.write('Testing, average recall: {:.4f}, average precision: {:.4f}, average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
    r, p, F1, acc, AUC
))
f.close()

model = model.cpu()
fig, axes = plt.subplots(2, 4)
negatives = 0
positives = 0
for i in range(len(testset)):
    if negatives == 4 and positives == 4:
        break
    image, label = testset[i]['img'], testset[i]['label']
    if label == 0 and negatives < 4:
        gradcam(model, image, detransformer, label, axes[0][negatives])
        negatives = negatives + 1
    elif label == 1 and positives < 4:
        gradcam(model, image, detransformer, label, axes[1][positives])
        positives = positives + 1

fig.savefig('fine-tuning_gradcam.png')
fig.show()

torch.save(model.state_dict(), 'fine-tune.pt')
