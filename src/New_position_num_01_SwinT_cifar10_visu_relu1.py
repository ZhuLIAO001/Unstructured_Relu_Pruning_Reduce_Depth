#calculate the entropy for relu1 (1st Conv2d layer inside each basic block) of resnet18 models


import numpy as np
import math
import torch
import torchvision
import torch.nn.utils.prune
#from pytorchcv.model_provider import get_model as ptcv_get_model # model
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
import wandb
from tqdm import tqdm
from torchvision.models import resnet18
import glob
import os
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import math
import matplotlib.ticker
from copy import deepcopy

## Choose device
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0")



###########################################
# CIFAR-10 dataset
num_classes = 10
batch_size = 1000

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset = torchvision.datasets.CIFAR10(root='~/data/cifar-10/',
											train=True, 
											transform=transform,
											download=True)
test_dataset = torchvision.datasets.CIFAR10(root='~/data/cifar-10/',
											train=False, 
											transform=transform,
											download=True)

## Dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										batch_size=batch_size, 
										shuffle=True,
										num_workers=8)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										batch_size=batch_size, 
										shuffle=True,
										num_workers=8)



class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)
		else:
			self.hook = module.register_backward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		self.output = torch.heaviside(torch.mean(torch.stack(list(input), dim=0),dim=0) ,torch.tensor([0],dtype=torch.float32).to(device))
	def close(self):
		self.hook.remove()

class Hook_Identity():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)
		else:
			self.hook = module.register_backward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		# self.output = torch.heaviside(torch.mean(torch.stack(list(input), dim=0),dim=0) ,torch.tensor([0],dtype=torch.float32).to(device))
		self.output =   torch.where(torch.mean(torch.stack(list(input), dim=0),dim=0)==0, 0, torch.ones_like(torch.mean(torch.stack(list(input), dim=0),dim=0))).to(device)
	def close(self):
		self.hook.remove()




def test(model):
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





# g = os.walk(r'/home/ids/ipp-9236/PYZhu/NIPS23/VGG16_cifar10/RELU_modelntropy_lamda_sweep/lamda_0.1_1_counInfiEntro_doubleBeta10000/finetune_acti_to_linear_model/lamda0.1/layer_f42')                                     #load the models
# g = os.walk(r'/home/ids/ipp-9236/PYZhu/ICIP23/SwinT_cifar10/model/remove_ZeroLayer_LinerActi_model') 
layer_out_num = []

# for root, dir_list, file_list in g:
# 	for file_name in file_list:
# 		print('file name',file_name)

results = []
results_layer = {}
num_filter_layer = {}
num_filter_each_layer = {}
# target_model= torch.load('/home/ids/ipp-9236/PYZhu/ICIP23/SwinT_cifar10/model/model_save/'+file_name).to(device)          #load the models
target_model= torch.load('/home/ids/ipp-9236/PYZhu/ICIP23/SwinT_cifar10/ReInitialize/model/sparsity_0.9375').to(device)  
l = 0
target_model.eval()
# print(target_model)


for name, module in target_model.named_modules():
	if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
		if 'downsample' in name:
			layer_out_num = layer_out_num
		else:
			layer_out_num.append(module.weight.data.shape[0])



# for name, module in target_model.named_modules():
# 	if type(module) == torch.nn.GELU or type(module) == torch.nn.Identity:
# 		num_filter_layer[name] = layer_out_num[l]
# 		l += 1



hooks = {}
for name, module in target_model.named_modules():
	if type(module) == torch.nn.GELU:
		hooks[name] = Hook(module, backward=False)
	if type(module) == torch.nn.Identity:
		hooks[name] = Hook_Identity(module, backward=False)



layer_len = {}
layer_name = []
layer_act_0_per = []
layer_act_1_per = []
layer_act_0_num = []
layer_act_1_num = []
layer_act_mix_num = []



with torch.no_grad():
	target_model.eval()
	correct=0
	total=0
	
	joint_activation_logger = {}
	entorpy0_position = {} 
																									
	for key in hooks.keys():                                  # relu, layer1.0.relu, layer1.1.relu, layer2.0.relu ...
		joint_activation_logger[key] = {}
		entorpy0_position[key] = {} 
		# for j in range(num_filter_layer[key]):
		# 	joint_activation_logger[key][j] = 0

	for key in entorpy0_position.keys():
		layer_name.append(key)
		entorpy0_position[key][0] = {}
		entorpy0_position[key][1] = {}

	class_counter = np.zeros(num_classes)
	with torch.no_grad():
		for data in tqdm(train_loader):
			images,labels=data[0].to(device),data[1].to(device)  
			outputs=target_model(images)             	
			for key in joint_activation_logger.keys():
				p_one = torch.mean(hooks[key].output,dim=0)
				# print(p_one.shape)                 	#([64, 16, 16])	
				while len(p_one.shape) > 1:       			
					p_one = torch.mean(p_one,dim=0)
				for j in range(p_one.shape[0]):
					num_filter_layer[key] = p_one.shape[0]
					joint_activation_logger[key][j] = 0
			break;





	class_counter = np.zeros(num_classes)
	with torch.no_grad():
		for data in tqdm(train_loader):
			images,labels=data[0].to(device),data[1].to(device)  
			outputs=target_model(images)             	
			for key in joint_activation_logger.keys():
				p_one = torch.mean(hooks[key].output,dim=0)
				print(p_one.shape)                 	#([64, 16, 16])	
				while len(p_one.shape) > 1:       			
					p_one = torch.mean(p_one,dim=0)
				for j in range(p_one.shape[0]):
					joint_activation_logger[key][j] += p_one[j].item()




	for key in joint_activation_logger.keys(): 
		total_act_0 = 0
		total_act_1 = 0
		for m in range(0, num_filter_layer[key]):
			if joint_activation_logger[key][m]/len(train_loader) ==0:
				total_act_0 += 1
				entorpy0_position[key][0][m] = 1
			if joint_activation_logger[key][m]/len(train_loader) ==1:
				total_act_1 += 1
				entorpy0_position[key][1][m] = 1
		
		act_0_per = total_act_0/num_filter_layer[key]
		act_1_per = total_act_1/num_filter_layer[key]
		act_mix_num = num_filter_layer[key] - total_act_0 - total_act_1

		layer_act_0_per.append(act_0_per)
		layer_act_0_num.append(total_act_0)
		layer_act_1_per.append(act_1_per) 
		layer_act_1_num.append(total_act_1)
		# print('layer_act_1_num',layer_act_1_num)
		layer_act_mix_num.append(act_mix_num) 

test_acc, test_loss = test(target_model)



# np.savez('/home/ids/ipp-9236/PYZhu/NIPS23/VGG16_cifar10/RELU_modelntropy_lamda_sweep/lamda_0.1_1_counInfiEntro_doubleBeta10000/finetune_prun_model/lamda0.1/para_per_num/para_position_01/' +'entropy_num_per'+file_name ,                 #save the parameters
# 			layer_name=layer_name, layer_act_0_per=layer_act_0_per, layer_act_1_per = layer_act_1_per, 
# 			layer_act_0_num = layer_act_0_num, layer_act_1_num=layer_act_1_num,layer_act_mix_num=layer_act_mix_num,entorpy0_position=entorpy0_position)

np.savez('/home/ids/ipp-9236/PYZhu/ICIP23/SwinT_cifar10/ReInitialize/para_position_01/' +'entropy_num_per'+'sparsity_0.9375' ,                 
			layer_name=layer_name, layer_act_0_per=layer_act_0_per, layer_act_1_per = layer_act_1_per, 
			layer_act_0_num = layer_act_0_num, layer_act_1_num=layer_act_1_num,layer_act_mix_num=layer_act_mix_num,entorpy0_position=entorpy0_position)




fig = plt.figure(figsize= (20,10),dpi = 500)        

p1 = plt.bar(layer_name, layer_act_0_num, width=0.4, color='darkorange', label='Nonlinear')    
p2 = plt.bar(layer_name, layer_act_1_num, width=0.4, color='indianred', bottom=layer_act_0_num, label='linear')
p3 = plt.bar(layer_name, layer_act_mix_num, width=0.4, color='royalblue', bottom=list(np.add(layer_act_0_num,layer_act_1_num)), label='mix')        
plt.title('number of neurons always linear/nonlinear with acc:'+ str(test_acc))                                                  
plt.legend() 

plt.legend([r'OFF ($\mathcal{H}(l,i|\Xi)=0$)', r'ON ($\mathcal{H}(l,i|\Xi)=0$)', r'$\mathcal{H}(l,i|\Xi) \neq 0$'], fontsize=50) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Layer index", fontsize=30)
plt.ylabel("Neurons", fontsize=30)
# plt.xticks(size=8)
plt.yticks(size=20)

plt.bar_label(p1, label_type='center')
plt.bar_label(p2, label_type='center')
plt.bar_label(p3, label_type='center')



# fig = plt.figure(figsize= (20,10),dpi = 500)        

# p1 = plt.bar(layer_name, layer_act_0_num, width=0.4, color='darkorange', label='Nonlinear')    
# p2 = plt.bar(layer_name, layer_act_1_num, width=0.4, color='indianred', bottom=layer_act_0_num, label='linear')
# p3 = plt.bar(layer_name, layer_act_mix_num, width=0.4, color='royalblue', bottom=list(np.add(layer_act_0_num,layer_act_1_num)), label='mix')        
# # plt.title('number of neurons always linear/nonlinear with acc:'+ str(test_acc))                                                  
# plt.legend() 

# plt.legend(['OFF (H=0)', 'ON (H=0)', r'${H} \neq 0$'], fontsize=40) 
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("Layer", fontsize=30)
# plt.ylabel("Neurons", fontsize=30)
# plt.xticks([])
# plt.yticks(size=20)



# plt.savefig('/home/ids/ipp-9236/PYZhu/ICIP23/SwinT_cifar10/model/para_per_num_remove_ZeroLayer_LinerActi/fig/' +'entro_per_num'+file_name  + '.pdf')            # save the bar plots for resnet model

plt.savefig('/home/ids/ipp-9236/PYZhu/ICIP23/SwinT_cifar10/ReInitialize/fig/' +'entro_per_num' +'sparsity_0.9375'+ '.pdf')



exit(0)



