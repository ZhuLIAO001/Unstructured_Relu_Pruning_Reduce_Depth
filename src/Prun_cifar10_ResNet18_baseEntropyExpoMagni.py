# Implement EGP for Resnet18 on Cifar10 dataset.

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import optim
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import random
import wandb
import os
from torch.utils.data import DataLoader

# set random seed, make results reproduceable
torch.manual_seed(43)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
random.seed(43)
np.random.seed(43)
torch.use_deterministic_algorithms(True)

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda:0')  


# project name on Wandb
project_name = "ICIP_resnet18_cifar10_baseEntropyExpoMagni_prun_" 





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



def test_entropy(model, hooks):
	model.eval()

	layers_entropy = {}
	entropy = {}
	for key in hooks.keys():
		entropy[key] = 0
	
	running_loss=0
	correct=0
	total=0    
	loss_fn=torch.nn.CrossEntropyLoss()


	with torch.no_grad():

		for data in tqdm(train_loader):
			images,labels=data[0].to(device),data[1].to(device)
			outputs=model(images)
			loss= loss_fn(outputs,labels)   
			running_loss+=loss.item()     
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()  


			for key in hooks.keys():         # For different layers	

				full_p_one = torch.heaviside(hooks[key].output , torch.tensor([0],dtype=torch.float32).to(device))
				p_one = torch.mean(full_p_one, dim=0)     
				state = hooks[key].output > 0                                       
				state = state.reshape(state.shape[0], state.shape[1], -1)                    
				state_sum = torch.mean(state*1.0 , dim=[0,2])                         
				state_sum_num = torch.sum((state_sum!= 0) * (state_sum!= 1))
				if state_sum_num != 0:
					while len(p_one.shape) > 1:					
						p_one = torch.mean(p_one,dim=1)
					p_one = (p_one*(state_sum!= 0) * (state_sum!= 1)*1.0)
					entropy[key] -= torch.sum(p_one*torch.log2(torch.clamp(p_one, min=1e-5))+((1-p_one)*torch.log2(torch.clamp(1-p_one, min=1e-5))))/state_sum_num					
				else:
					entropy[key] -= 0

	for key in hooks.keys():		
		layers_entropy[key] = entropy[key] / len(train_loader)

	test_loss=running_loss/len(train_loader)
	accu=100.*correct/total
	

	print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
	return(accu, test_loss, layers_entropy)




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


# load the modified model
from cifar10_models.resnet import resnet18
model = resnet18()
model.to(device)


hooks = {}
for name, module in model.named_modules():
	if type(module) == torch.nn.ReLU:
		hooks[name] = Hook(module)



sparsity_curve=[]
acc_curve=[]


sparsity = 0
sparsity_curve.append(sparsity)

name_of_run = '_sparsity_'+str(sparsity)
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
	torch.save(temp_model, 'YOUR PATH'+ '/Resnet18_cifar10/baseline_prun/model/'+ name_model)                   #set your own path to save model

acc_curve.append(final_testacc)


wandb.finish()






for i in range(1, 10):
	sparsity = 1-(1-fixed_amount_of_pruning)**i
	sparsity_curve.append(sparsity)

	name_of_run = 'sparsity_'+str(sparsity)
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
	

	layers_to_prune=[]
	for key in hooks.keys():
		if key == 'relu':
			layers_to_prune.append('conv1')
		else:
			name = key.replace('relu', 'conv')
			layers_to_prune.append(name)
  




	test_entropy_acc, test_entropy_loss, layers_entropy = test_entropy(model,hooks)   #calculate the entropy for each layer

	for key in hooks.keys():
		if key == 'relu':
			layers_entropy['conv1'] = layers_entropy.pop("relu")
		else:
			name = key.replace('relu', 'conv')
			layers_entropy[name] = layers_entropy.pop(key)



	#delete the entropy 0 layer in prun list
	for key in layers_entropy.keys():
		if layers_entropy[key] == 0:
			layers_to_prune.remove(key)
			
	
	layer_entro_magni = {}
	for name, module in model.named_modules():
		if name in layers_to_prune:
			if torch.numel(torch.abs(module.weight)[module.weight != 0])==0:
				layers_to_prune.remove(name)
			else:
				layer_entro_magni[name] = layers_entropy[name] * torch.mean(torch.abs(module.weight[module.weight!=0]))


	total_layers_entro_magni = 0
	
	for key in layer_entro_magni.keys():
		total_layers_entro_magni += layer_entro_magni[key].item()


	entropy_magni_layer_head = {}
	entropy_layer_head_expo = {}


	total_layers_weight_paras = 0
	for name, module in model.named_modules():
		if name in layers_to_prune:
			total_layers_weight_paras += torch.numel(module.weight[module.weight!=0])

	total_layers_weight_paras_to_prun = fixed_amount_of_pruning * total_layers_weight_paras


	wait_distri_paras = copy.deepcopy(layers_to_prune)
	fix_prun_amount={}
	left_amount={}


	while True:
		amout_changed = False
		total_entropy_layer_head_expo  = 0
		entropy_magni_layer_head = {}

		for name, module in model.named_modules():
			if name in wait_distri_paras:
				entropy_magni_layer_head[name]  = total_layers_entro_magni/(layer_entro_magni[name])
		
		max_value_entropy_magni_layer_head = max(entropy_magni_layer_head.values())
		for name, module in model.named_modules():
			if name in wait_distri_paras:
				entropy_layer_head_expo[name]  = torch.exp(entropy_magni_layer_head[name] - max_value_entropy_magni_layer_head).item()
				total_entropy_layer_head_expo += entropy_layer_head_expo[name]

		for name, module in model.named_modules():
			if name in wait_distri_paras:
				fix_prun_amount[name] = (total_layers_weight_paras_to_prun * (entropy_layer_head_expo[name]/total_entropy_layer_head_expo))
		
		for name, module in model.named_modules():
			if name in wait_distri_paras:

				left_amount[name] = torch.numel(module.weight[module.weight!=0])

				if left_amount[name] < fix_prun_amount[name]:
					fix_prun_amount[name] = left_amount[name]
					total_layers_weight_paras_to_prun -= left_amount[name]
					total_layers_entro_magni -= layer_entro_magni[name]			
					wait_distri_paras.remove(name)
					amout_changed = True

		if not amout_changed:
			break

	for name, module in model.named_modules():
		if name in layers_to_prune:
			prune.l1_unstructured(module, name='weight', amount=int(fix_prun_amount[name]))

	true_sparsity_weights = []
	zero_weight = 0
	whole_weight = 0
	for name, module in model.named_modules():
		if name in layers_to_prune:
			true_sparsity_weights.append(torch.numel(module.weight[module.weight==0])/torch.numel(module.weight))
			zero_weight += torch.numel(module.weight[module.weight==0])
			whole_weight += torch.numel(module.weight)
			print(torch.numel(module.weight[module.weight==0]))

	wandb.config.True_sparsity_weights = true_sparsity_weights
	global_sparsity = round(zero_weight/whole_weight, 2)

	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
	loss_fn=nn.CrossEntropyLoss().to(device)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

	final_testacc = 0

	for epoch in range(1,epochs+1):
		train_acc, train_loss = train(model, epoch,optimizer)
		test_acc, test_loss = test(model, epoch)
		final_testacc = test_acc
		last_lr=scheduler.get_last_lr()[-1]
		scheduler.step()
		wandb.log(
			{"train_acc": train_acc, "train_loss": train_loss,
			"test_acc": test_acc, "test_loss": test_loss, 
			'lr':last_lr, 'global_sparsity':global_sparsity})
		
		temp_model = copy.deepcopy(model)
		for name, module in temp_model.named_modules():
			if name in layers_to_prune:
				prune.remove(module,'weight')

		torch.save(temp_model, 'YOUR PATH'+ '/Resnet18_cifar10/baseEntropyExpoMagni_prun/model/'+ name_model)                              #set your own path to save model

	acc_curve.append(final_testacc)

	wandb.finish()


	fig = plt.figure(figsize= (20,10),dpi = 100)
	plt.plot(sparsity_curve, acc_curve, c='blue')
	for i in range(len(sparsity_curve)):
		plt.text(sparsity_curve[i], acc_curve[i], round(acc_curve[i],2))
	plt.scatter(sparsity_curve, acc_curve, c='red')
	plt.grid(True, linestyle='--', alpha=0.5)

	plt.xlabel("Parameters Pruned away", fontdict={'size': 16})
	plt.ylabel("modle_acc", fontdict={'size': 16})
	plt.title("Trade-off curve", fontdict={'size': 20})
	plt.savefig('YOUR PATH'+ '/Resnet18_cifar10/baseEntropyExpoMagni_prun/Tradeoff_curve/'+'sparsity_acc_Tradeoff_curve_'+str(sparsity) + '.pdf')          #set your own path to save trade-off figure


