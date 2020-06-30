import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import pandas as pd
import random 
from shutil import copyfile
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data import Augmentor


from torchvision import transforms, utils
from sklearn.metrics import roc_auc_score
from skimage.io import imread, imsave
import skimage

torch.cuda.empty_cache()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
	transforms.RandomHorizontalFlip(),
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




batchsize=10

if __name__ == '__main__':
	trainset = CovidCTDataset(root_dir='../../Images',
							  txt_COVID='../../Data-split/COVID/trainCT_COVID.txt',
							  txt_NonCOVID='../../Data-split/NonCOVID/trainCT_NonCOVID.txt',
							  transform=train_transformer)
	valset = CovidCTDataset(root_dir='../../Images',
							  txt_COVID='../../Data-split/COVID/valCT_COVID.txt',
							  txt_NonCOVID='../../Data-split/NonCOVID/valCT_NonCOVID.txt',
							  transform=val_transformer)
	testset = CovidCTDataset(root_dir='../../Images',
							  txt_COVID='../../Data-split/COVID/testCT_COVID.txt',
							  txt_NonCOVID='../../Data-split/NonCOVID/testCT_NonCOVID.txt',
							  transform=val_transformer)
	print(trainset.__len__())
	print(valset.__len__())
	print(testset.__len__())

	train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
	val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
	test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)


alpha = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(optimizer, epoch):
	
	model.train()
	
	train_loss = 0
	train_correct = 0
	
	for batch_index, batch_samples in enumerate(train_loader):
		
		# move data to device
		data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
		
		
		optimizer.zero_grad()
		output = model(data)
		
		criteria = nn.CrossEntropyLoss()
		loss = criteria(output, target.long())
		train_loss += criteria(output, target.long())
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		pred = output.argmax(dim=1, keepdim=True)
		train_correct += pred.eq(target.long().view_as(pred)).sum().item()
	
		# Display progress
		if batch_index % bs == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
				epoch, batch_index, len(train_loader),
				100.0 * batch_index / len(train_loader), loss.item() / bs))
	
	print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
		100.0 * train_correct / len(train_loader.dataset)))
	f = open('model_result/{}.txt'.format(modelname), 'a+')
	f.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
		100.0 * train_correct / len(train_loader.dataset)))
	f.write('\n')
	f.close()


def val(epoch):
	
	model.eval()
	
	# Don't update model
	with torch.no_grad():

		predlist=[]
		scorelist=[]
		targetlist=[]
		# Predict
		for batch_index, batch_samples in enumerate(val_loader):
			data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
			output = model(data)
			
			score = F.softmax(output, dim=1)
			pred = output.argmax(dim=1, keepdim=True)

			targetcpu=target.long().cpu().numpy()
			predlist=np.append(predlist, pred.cpu().numpy())
			scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
			targetlist=np.append(targetlist, targetcpu)
		   		  
	return targetlist, scorelist, predlist


def test(epoch):
	
	model.eval()
	
	# Don't update model
	with torch.no_grad():
		
		predlist=[]
		scorelist=[]
		targetlist=[]
		# Predict
		for batch_index, batch_samples in enumerate(test_loader):
			data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
			output = model(data)
			
			score = F.softmax(output, dim=1)
			pred = output.argmax(dim=1, keepdim=True)

			targetcpu=target.long().cpu().numpy()
			predlist=np.append(predlist, pred.cpu().numpy())
			scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
			targetlist=np.append(targetlist,targetcpu)
		   
	return targetlist, scorelist, predlist


# In[46]:


# %CheXNet pretrain
# class DenseNet121(nn.Module):
#	  """Model modified.

#	  The architecture of our model is the same as standard DenseNet121
#	  except the classifier layer which has an additional sigmoid function.

#	  """
#	  def __init__(self, out_size):
#		  super(DenseNet121, self).__init__()
#		  self.densenet121 = torchvision.models.densenet121(pretrained=True)
#		  num_ftrs = self.densenet121.classifier.in_features
#		  self.densenet121.classifier = nn.Sequential(
#			  nn.Linear(num_ftrs, out_size),
#			  nn.Sigmoid()
#		  )

#	  def forward(self, x):
#		  x = self.densenet121(x)
#		  return x
  

# device = 'cuda'
# CKPT_PATH = 'model.pth.tar'
# N_CLASSES = 14

# DenseNet121 = DenseNet121(N_CLASSES).cuda()

# CKPT_PATH = './CheXNet/model.pth.tar'

# if os.path.isfile(CKPT_PATH):
#	  checkpoint = torch.load(CKPT_PATH)		
#	  state_dict = checkpoint['state_dict']
#	  remove_data_parallel = False


#	  pattern = re.compile(
#		  r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
#	  for key in list(state_dict.keys()):
#		  match = pattern.match(key)
#		  new_key = match.group(1) + match.group(2) if match else key
#		  new_key = new_key[7:] if remove_data_parallel else new_key
#		  new_key = new_key[7:]
#		  state_dict[new_key] = state_dict[key]
#		  del state_dict[key]


#	  DenseNet121.load_state_dict(checkpoint['state_dict'])
#	  print("=> loaded checkpoint")
# #		print(densenet121)
# else:
#	  print("=> no checkpoint found")

# # for parma in DenseNet121.parameters():
# #			parma.requires_grad = False
# DenseNet121.densenet121.classifier._modules['0'] = nn.Linear(in_features=1024, out_features=2, bias=True)
# DenseNet121.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# # print(DenseNet121)
# model = DenseNet121.to(device)


### DenseNet

class DenseNet169(nn.Module):

	def __init__(self):
		"""
		DenseNet169 with binary classification
		"""
		super(DenseNet169, self).__init__()
		self.dense_net = torchvision.models.densenet169(num_classes=2)

	def forward(self, x):
		logits = self.dense_net(x)
		return logits
	
model = DenseNet169().cuda() if torch.cuda.is_available() else DenseNet169()
modelname = 'DenseNet169'
# print(model)


# train
bs = 10
votenum = 10
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []
vote_pred = np.zeros(valset.__len__())
vote_score = np.zeros(valset.__len__())

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

total_epoch = 3000
for epoch in range(1, total_epoch + 1):
	train(optimizer, epoch)
	
	targetlist, scorelist, predlist = val(epoch)
	vote_pred = vote_pred + predlist 
	vote_score = vote_score + scorelist 

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
		

		torch.save(model.state_dict(), "model_backup/{}.pt".format(modelname))	
		
		vote_pred = np.zeros(valset.__len__())
		vote_score = np.zeros(valset.__len__())
		print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
		epoch, r, p, F1, acc, AUC))

		f = open('model_result/{}.txt'.format(modelname), 'a+')
		f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
		epoch, r, p, F1, acc, AUC))
		f.close()


# test
bs = 10
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []
vote_pred = np.zeros(testset.__len__())
vote_score = np.zeros(testset.__len__())

total_epoch = 10
for epoch in range(1, total_epoch+1):
	
	targetlist, scorelist, predlist = test(epoch)
	vote_pred = vote_pred + predlist 
	vote_score = vote_score + scorelist 
	
	TP = ((predlist == 1) & (targetlist == 1)).sum()
	TN = ((predlist == 0) & (targetlist == 0)).sum()
	FN = ((predlist == 0) & (targetlist == 1)).sum()
	FP = ((predlist == 1) & (targetlist == 0)).sum()

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
	print('AUC', AUC)

	if epoch % votenum == 0:
		
		# major vote
		vote_pred[vote_pred <= (votenum/2)] = 0
		vote_pred[vote_pred > (votenum/2)] = 1
		
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
		print('AUC', AUC)
		
		
		vote_pred = np.zeros((1,testset.__len__()))
		vote_score = np.zeros(testset.__len__())
		print('vote_pred',vote_pred)
		print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
		epoch, r, p, F1, acc, AUC))

		f = open('model_result/test_{modelname}.txt', 'a+')
		f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
		epoch, r, p, F1, acc, AUC))
		f.close()
