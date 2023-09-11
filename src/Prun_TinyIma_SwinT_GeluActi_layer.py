#  Traditional iterative pruning of Swin-T (only prun GeLU activated layers) on Cifar10 dataset.

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

# set random seeds, make results reproduceable
torch.manual_seed(43)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
random.seed(43)
np.random.seed(43)
torch.use_deterministic_algorithms(True)

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = torch.device('cuda:0')  


# project name on Wandb
project_name = "ICIP_SwinT_TinyIma_baseline_prun_GeluActi" 


# Training parameters 
epochs = 160
learning_rate = 0.001
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




DATA_DIR = '/data/datasets/tiny-imagenet-200'                     # set your own dataset path

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')

# Define transformation sequence for image pre-processing
transform_train = transforms.Compose([
									  transforms.RandomHorizontalFlip(),
									  transforms.ToTensor(),
									  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = transforms.Compose([
									 transforms.ToTensor(),
									 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform_train)

train_loader = DataLoader(train_dataset,
						  batch_size=batch_size, 
						  shuffle=True, 
						  num_workers=8)


test_dataset = torchvision.datasets.ImageFolder(VALID_DIR, transform=transform_test)
test_loader = DataLoader(test_dataset,
						  batch_size=batch_size, 
						  shuffle=False, 
						  num_workers=8)




model = torchvision.models.swin_t( weights = True)
model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=1000)
model.to(device)


hooks = {}
for name, module in model.named_modules():
	if type(module) == torch.nn.GELU:
		hooks[name] = Hook(module)



sparsity_curve=[]
acc_curve=[]

sparsity = 0
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
	torch.save(temp_model, 'YOUR PATH'+ '/SwinT_TinyImagi/baseline_GeluActivated/model/model_save/'+ name_model)                   #set your own path to save model

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

	previous_conv_name = None
	for name, module in model.named_modules():
		if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
			previous_conv_name = name
		elif name in hooks.keys():
			layers_to_prune.append(previous_conv_name)
					
	prune_list = []
	for name, module in model.named_modules():
		if name in layers_to_prune:
			prune_list.append((module, "weight"))


	torch.nn.utils.prune.global_unstructured(prune_list, pruning_method=torch.nn.utils.prune.L1Unstructured, amount=fixed_amount_of_pruning)

	true_sparsity_weights = []
	zero_weight = 0
	whole_weight = 0
	for name, module in model.named_modules():
		if name in layers_to_prune:
			true_sparsity_weights.append(torch.numel(module.weight[module.weight==0])/torch.numel(module.weight))
			zero_weight += torch.numel(module.weight[module.weight==0])
			whole_weight += torch.numel(module.weight)

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
		


		torch.save(model, 'YOUR PATH'+ '/SwinT_TinyImagi/baseline_GeluActivated/model/check_point/'+'check_point_'+ name_model)             #set your own path to save check point


	
	acc_curve.append(final_testacc)

	temp_model = copy.deepcopy(model)
	for name, module in temp_model.named_modules():
		if name in layers_to_prune:
			prune.remove(module,'weight')

	torch.save(temp_model, 'YOUR PATH'+ '/SwinT_TinyImagi/baseline_GeluActivated/model/model_save/'+ name_model)             #set your own path to save model

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



	plt.savefig('YOUR PATH'+ '/SwinT_TinyImagi/baseline_GeluActivated/Tradeoff_curve/'+' sparsity_acc_Tradeoff_curve_'+str(sparsity) + '.pdf')              #set your own path to save trade-off figure
	# plt.show()