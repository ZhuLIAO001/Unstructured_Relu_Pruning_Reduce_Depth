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
# import time

# time.sleep(3*60*60)

torch.manual_seed(43)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
random.seed(43)
np.random.seed(43)
torch.use_deterministic_algorithms(True)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda:0')  # Device configuration


# Hyper-parameters
project_name = "ICIP_SwinT_cifar10_baseline_prun" 
# project_name = "ICIP_text_" 



# model_name = './pruned_SDD_TinyIMAGENET_ResNet' # name of saved dense model

# path = '/models/SDD'
# name_of_run = 'ResNet50_TinyIMAGENET_Prun_512'

# Training parameters 
epochs = 160
learning_rate = 0.001
momentum = 0.9
gamma=0.1
weight_decay = 1e-4
milestones=[80,120]
batch_size = 128
fixed_amount_of_pruning = 0.5



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
		loss.backward(retain_graph=True)
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



# model = torchvision.models.swin_t(weights = True)
# model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=10)
# model.to(device)


model= torch.load('/mnt/data/Zhu/ICIP23/SwinT_cifar10/baseline/model/check_point/check_point_sparsity_0.9375').to(device)



sparsity_curve=[]
acc_curve=[]



# sparsity = 0
# sparsity_curve.append(sparsity)

# # test_acc, test_loss = test(model, 1)
# # acc_curve.append(test_acc)
# name_of_run = 'sparsity_'+str(sparsity)
# name_model = name_of_run

# wandb.init(project=project_name, entity="zhu-liao")
# wandb.run.name = name_of_run
# wandb.config.epochs = epochs
# wandb.config.batch_size = batch_size
# wandb.config.learning_rate = learning_rate
# wandb.config.weight_decay = weight_decay
# wandb.config.gamma = gamma
# wandb.config.milestones = milestones
# wandb.config.momentum = momentum
# wandb.config.sparsity = sparsity


# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
# loss_fn=nn.CrossEntropyLoss().to(device)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# final_testacc = 0

# for epoch in range(1,epochs+1):
# 	train_acc, train_loss = train(model, epoch, optimizer)
# 	test_acc, test_loss = test(model, epoch)
# 	final_testacc = test_acc
# 	last_lr=scheduler.get_last_lr()[-1]
# 	scheduler.step()
# 	wandb.log(
# 		{"train_acc": train_acc, "train_loss": train_loss,
# 		"test_acc": test_acc, "test_loss": test_loss, 
# 		'lr':last_lr, 'global_sparsity':0})


# 	temp_model = copy.deepcopy(model)
# 	torch.save(temp_model, '/home/ipp-9236/PYZhu/ICIP23/SwinT_cifar10/baseline/model/model_save/'+ name_model)

# acc_curve.append(final_testacc)


# wandb.finish()





for i in range(5, 10):
	sparsity = 1-(1-fixed_amount_of_pruning)**i
	sparsity_curve.append(sparsity)

	name_of_run = 'sparsity_'+str(sparsity)
	name_model = name_of_run

	wandb.init(project=project_name, entity="zhu-liao")
	wandb.run.name = name_of_run
	wandb.config.epochs = epochs
	wandb.config.batch_size = batch_size
	wandb.config.learning_rate = learning_rate
	wandb.config.weight_decay = weight_decay
	wandb.config.gamma = gamma
	wandb.config.milestones = milestones
	wandb.config.momentum = momentum
	wandb.config.sparsity = sparsity
	


			
	layers_to_prune = [(module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear, model.modules())]	
	wandb.config.pruned_layers = layers_to_prune

	torch.nn.utils.prune.global_unstructured(layers_to_prune, pruning_method=torch.nn.utils.prune.L1Unstructured, amount=fixed_amount_of_pruning)	



	
	true_sparsity_weights = []
	zero_weight = 0
	whole_weight = 0
	for name, module in model.named_modules():
		if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
			true_sparsity_weights.append(torch.numel(module.weight[module.weight==0])/torch.numel(module.weight))
			zero_weight += torch.numel(module.weight[module.weight==0])
			whole_weight += torch.numel(module.weight)
			# print(name)
			# print(torch.numel(module.weight[module.weight==0]))

	wandb.config.True_sparsity_weights = true_sparsity_weights

	global_sparsity = round(zero_weight/whole_weight, 2)



	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
	loss_fn=nn.CrossEntropyLoss().to(device)

	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

	final_testacc = 0

	# for epoch in range(1,epochs+1):
	# # for epoch in range(1,2):
	# 	train_acc, train_loss = train(model, epoch, optimizer)
	# 	test_acc, test_loss = test(model, epoch)
	# 	final_testacc = test_acc
	# 	last_lr=scheduler.get_last_lr()[-1]
	# 	scheduler.step()
	# 	wandb.log(
	# 		{"train_acc": train_acc, "train_loss": train_loss,
	# 		"test_acc": test_acc, "test_loss": test_loss, 
	# 		'lr':last_lr, 'global_sparsity':0})



	# 	torch.save(model, '/mnt/data/Zhu/ICIP23/SwinT_cifar10/baseline/model/check_point/'+'check_point_'+ name_model)


	
	acc_curve.append(final_testacc)

	temp_model = torch.load('/mnt/data/Zhu/ICIP23/SwinT_cifar10/baseline/model/check_point/'+'check_point_'+ name_model).to(device)
	for name, module in temp_model.named_modules():
		if name in layers_to_prune:
			prune.remove(module,'weight')

	torch.save(temp_model, '/mnt/data/Zhu/ICIP23/SwinT_cifar10/baseline/model/model_save/'+ name_model)

	wandb.finish()




	fig = plt.figure(figsize= (20,10),dpi = 500)

	plt.plot(sparsity_curve, acc_curve, '-',c='cornflowerblue',linewidth=5)
	for i in range(len(sparsity_curve)):
		plt.text(sparsity_curve[i], acc_curve[i], round(acc_curve[i],2), fontdict={'size': 15})
	plt.scatter(sparsity_curve, acc_curve, c='red')


	plt.grid(True, linestyle='--', alpha=0.5)

	plt.xlabel(r"Sparsity $\zeta$", fontdict={'size': 30})
	plt.ylabel("Accuracy (%)", fontdict={'size': 30})
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)



	plt.savefig('/mnt/data/Zhu/ICIP23/SwinT_cifar10/baseline/Tradeoff_curve/'+' sparsity_acc_Tradeoff_curve_'+str(sparsity) + '.png')
	# plt.show()

