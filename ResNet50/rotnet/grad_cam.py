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


device = 'cuda'

#labels are reversed: COVID_positive is actually COVID_negative
#some goes for COVID_negative which is actually COVID_positive
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

def generate_label_and_predicted_label(label, output_batch):
	if rotate:
		label = label_rotation(label)
		pred = label_rotation(output_batch.argmax(dim=1, keepdim=True)[0].item())
	else:
		label = label_covid(label)
		pred = label_covid(output_batch.argmax(dim=1, keepdim=True)[0].item())
	return label, pred

def get_activations(name):
	def hook(module, input, output):
		activations[name] = output.detach()
	return hook

def get_grads(name):
	def hook(module, grad_input, grad_output):
		grads[name] = grad_output # is a tuple
	return hook

def get_image_and_label(batch_samples, rotate=False):
	if rotate:
		input_batch, label = rotateBatch(batch_samples['img'])
	else:
		input_batch = batch_samples['img']
		label = batch_samples['label']
	return input_batch, label

batchsize = 1
trainset, valset, testset = getTransformedDataSplit()
train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)

#path to model to which gradcam is supposed to be applied
path = 'model_backup/ResNet50_sslRotateNoPretrain_test_covid.pt'
grad_cam_path = 'gradcam/'+path.split('_')[2]+'/'
if not os.path.exists(grad_cam_path):
    os.makedirs(grad_cam_path)
model = resnet50()

#change number of classes according to model: sslRotate:4, Finetune:2
model.change_cls_number(num_classes=4)
model.load_state_dict(torch.load(path))

#set to False if model is Finetune
rotate = True

model.layer4.register_forward_hook(get_activations("layer4"))
model.layer4.register_backward_hook(get_grads("layer4"))

model.eval()
i = 0
for batch_index, batch_samples in tqdm(enumerate(test_loader)):
	activations = dict()
	grads = dict()

	model.cuda()
	input_batch, label = get_image_and_label(batch_samples, rotate)
	input_batch = input_batch.to(device)

	output_batch = model(input_batch)
	output_batch[0][0].backward()

	activations = activations["layer4"]
	grads = grads["layer4"][0]

	grads = torch.mean(grads, dim=(-2, -1), keepdims=True) # gradients are pooled
	grad_cam = activations * grads
	grad_cam = torch.sum(grad_cam, dim=-3) # weighted sum
	grad_cam = torch.nn.functional.relu(grad_cam) # relu'd weighted sum
	grad_cam = grad_cam / grad_cam.max() # normalization
	grad_cam = grad_cam[0] # squeezing the batch of one

	label, pred = generate_label_and_predicted_label(label.item(), output_batch)

	plt.imshow(input_batch[0,1,:,:].cpu().numpy(), alpha=1.0)
	plt.imshow(torchvision.transforms.ToPILImage()(grad_cam.cpu()).resize((224, 224), resample=Image.BILINEAR), cmap='jet', alpha=0.5)
	plt.title(label)
	plt.savefig(grad_cam_path+'grad_cam_'+'predicted'+pred+'_'+str(i)+'.png')
	i += 1
