import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import  DataLoader
from tqdm import tqdm
import os
from util.resNet import *
from util.dataloadersGeneration import *
from util.trainValTest import *
from util.trainValTest import *


def label_covid(label):
	if label:
		return 'COVID_positive'
	else:
		return 'COVID_negative'

def label_rotation(label):
	if label == 0:
		return '0°'
	elif label == 1:
		return '90°'
	elif label == 2:
		return '180°'
	else:
		return '270°'

def get_activations(name):
	def hook(module, input, output):
		activations[name] = output.detach()
	return hook

def get_grads(name):
	def hook(module, grad_input, grad_output):
		grads[name] = grad_output # is a tuple
	return hook

batchsize = 1
trainset, valset, testset = getTransformedDataSplit()
train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

path = 'model_backup/ResNet50_sslRotateNoPretrain_test_covid.pt'
grad_cam_path = 'gradcam/'+path.split('_')[2]+'/'
if not os.path.exists(grad_cam_path):
    os.makedirs(grad_cam_path)
model = resnet50()
model.change_cls_number(num_classes=4)
model.load_state_dict(torch.load(path))

model.layer4.register_forward_hook(get_activations("layer4"))
model.layer4.register_backward_hook(get_grads("layer4"))

model.eval()
i = 0
for batch_index, batch_samples in tqdm(enumerate(test_loader)):
	activations = dict()
	grads = dict()
	model.cuda()
	rotate = True
	if rotate:
		input_batch, label = rotateBatch(batch_samples['img'])
		input_batch = input_batch.to('cuda')
		# print(len(batch_samples))
		# for input_batch, label in zip(batch_samples['img'], batch_samples['label']):
	else:
	#print(input_batch.shape)
		label = batch_samples['label']
		input_batch = batch_samples['img'].to('cuda')

	# print(label)
	#print(input_batch.unsqueeze(0).shape)
	output_batch = model(input_batch)
	# print(output_batch.shape)
	#output_batch[0][671].backward() # 671 -- the ImageNet index for bikes
	#for pred, image, label in zip(output_batch, input_batch, batch_samples['label']):
	#print(pred[0])
	#print(output_batch.argmax(dim=1, keepdim=True)[0].item())
	output_batch[0][0].backward()

	activations = activations["layer4"]
	grads = grads["layer4"][0]

	# print(activations.shape)
	# print(grads.shape)
	grads = torch.mean(grads, dim=(-2, -1), keepdims=True) # gradients are pooled
	grad_cam = activations * grads
	grad_cam = torch.sum(grad_cam, dim=-3) # weighted sum
	grad_cam = torch.nn.functional.relu(grad_cam) # relu'd weighted sum
	grad_cam = grad_cam / grad_cam.max() # normalization
	grad_cam = grad_cam[0] # squeezing the batch of one

	if rotate:
		label = label_rotation(label.item())
		pred = label_rotation(output_batch.argmax(dim=1, keepdim=True)[0].item())
	else:
		label = label_covid(label.item())
		pred = label_covid(output_batch.argmax(dim=1, keepdim=True)[0].item())

	plt.imshow(input_batch[0,1,:,:].cpu().numpy(), alpha=1.0)
	plt.imshow(torchvision.transforms.ToPILImage()(grad_cam.cpu()).resize((224, 224), resample=Image.BILINEAR), cmap='jet', alpha=0.5)
	plt.title(label)
	plt.savefig(grad_cam_path+'grad_cam_'+'predicted'+pred+'_'+str(i)+'.png')
	i += 1
