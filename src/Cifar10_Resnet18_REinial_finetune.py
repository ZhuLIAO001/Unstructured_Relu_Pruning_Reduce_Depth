# We reinitialize EGP model (Resnet18 on Cifar10) and finetune it. To see can we successfully train from scratch a shallower model, without resorting to an iterative strategy?

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import copy
import random
import wandb
import os


# set random seeds, make results reproduceable
torch.manual_seed(43)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
random.seed(43)
np.random.seed(43)
torch.use_deterministic_algorithms(True)

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device('cuda:0')  


# project name on Wandb
project_name = "ICIP_Resnet18_Cifar10_ReInit" 



# Training parameters 
epochs = 160
learning_rate = 0.1
momentum = 0.9
gamma=0.1
weight_decay = 1e-4
milestones=[80,120]
batch_size = 128
fixed_amount_of_pruning = 0.5


class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)                        
		else:
			self.hook = module.register_backward_hook(self.hook_fn)													 
	def hook_fn(self, module, input, output):
		# self.output = output
		self.output = torch.mean(torch.stack(list(input), dim=0),dim=0)   
	def close(self):
		self.hook.remove()



def train(model, epoch, optimizer):
	print('\nEpoch : %d'%epoch)    
	model.train()
	
	running_loss=0
	correct=0
	total=0    
	loss_fn=torch.nn.CrossEntropyLoss()
	for data in tqdm(train_loader):       
		inputs,labels=data[0].to(device),data[1].to(device)        
		outputs=model(inputs)       
		loss=loss_fn(outputs,labels)       
		optimizer.zero_grad()
		loss.backward()
		# print('train loss:', loss)
		optimizer.step()     
		running_loss += loss.item()        
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
		
	train_loss=running_loss/len(train_loader)
	accu=100.*correct/total
		
	print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
	return(accu, train_loss)




def test(model, epoch):
	model.eval()
	
	running_loss=0
	correct=0
	total=0    
	loss_fn=torch.nn.CrossEntropyLoss()
	with torch.no_grad():
		for data in tqdm(test_loader):
			images,labels=data[0].to(device),data[1].to(device)
			outputs=model(images)
			loss= loss_fn(outputs,labels)
			running_loss+=loss.item()     
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()   
	test_loss=running_loss/len(test_loader)
	accu=100.*correct/total
	

	print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
	return(accu, test_loss)



# Image preprocessing modules
transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomCrop(32,4),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='~/data/cifar-10/',
											 train=True, 
											 transform=transform,
											 download=True)

test_dataset = torchvision.datasets.CIFAR10(root='~/data/cifar-10/',
											train=False, 
											transform=transforms.Compose([transforms.ToTensor(),
																		  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
											download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size, 
										   shuffle=True,
										   num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size, 
										  shuffle=False,
										  num_workers = 8)




model= torch.load('YOUR PATH' + '/Resnet18_cifar10/baseEntropyExpoMagni_prun/model/sparsity_0.99609375').to(device)                #Your path where save the EGP model


hooks = {}
for name, module in model.named_modules():
	if type(module) == torch.nn.ReLU:
		hooks[name] = Hook(module)




sparsity_curve=[]
acc_curve=[]


sparsity = 0.9375



# Reinilize the shallower model
for name, module in list(model.named_modules()):
	if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
		if torch.numel(torch.abs(module.weight)[module.weight != 0])==0:
			for param in module.parameters():
				param.requires_grad = False
		else:
			torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
	elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
		nn.init.constant_(module.weight, 1)
		nn.init.constant_(module.bias, 0)




#Finetune the reinitilized model
name_of_run = 'sparsity_'+str(0.996)
name_model = name_of_run

#wandb setting
wandb.init(project=project_name, entity="YOUR ENEITY")                                                #set your own entity
wandb.run.name = name_of_run
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.learning_rate = learning_rate
wandb.config.weight_decay = weight_decay
wandb.config.gamma = gamma
wandb.config.milestones = milestones
wandb.config.momentum = momentum
wandb.config.sparsity = sparsity


optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
loss_fn=nn.CrossEntropyLoss().to(device)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

final_testacc = 0

for epoch in range(1,epochs+1):
	train_acc, train_loss = train(model, epoch, optimizer)
	test_acc, test_loss = test(model, epoch)
	final_testacc = test_acc
	last_lr=scheduler.get_last_lr()[-1]
	scheduler.step()
	wandb.log(
		{"train_acc": train_acc, "train_loss": train_loss,
		"test_acc": test_acc, "test_loss": test_loss, 
		'lr':last_lr, 'global_sparsity':0})


	temp_model = copy.deepcopy(model)
	torch.save(temp_model, 'YOUR PATH'+ '/Resnet18_cifar10/baseEntropyExpoMagni_prun/ReInitialize/model/'+ name_model)	   	    #set your own path to save the finetuned model

acc_curve.append(final_testacc)


wandb.finish()

