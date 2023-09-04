# We reinitialize EGP model (Resnet50 on TinyImagenet) and finetune it. To see can we successfully train from scratch a shallower model, without resorting to an iterative strategy?

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import optim
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
os.environ["CUDA_VISIBLE_DEVICES"]="6"
device = torch.device('cuda:0')  # Device configuration


# project name on Wandb
project_name = "ICIP_Resnet50_TinyImag_ReInit" 



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





DATA_DIR = '/data/datasets/tiny-imagenet-200'                                # set your own dataset path

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




from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import ResNet

class Bottleneck_new(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(Bottleneck_new, self).__init__(*args, **kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        delattr(self, 'relu')
	

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out



class ResNet_new(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet_new, self).__init__(*args, **kwargs)

        
    def _replace_module(self, name, module):
        modules = name.split('.')
        curr_mod = self
        
        for mod_name in modules[:-1]:
            curr_mod = getattr(curr_mod, mod_name)
        
        setattr(curr_mod, modules[-1], module)



model= torch.load('YOUR PATH' + '/Resnet50_TinyImag/baseEntropyMagni_prun/model/model_save/sparsity_0.9375').to(device)               #Your path where save the EGP model


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




# Reinilize the shallower model
name_of_run = 'sparsity_'+str(0.9375)
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
	torch.save(temp_model, 'YOUR PATH' + '/Resnet50_TinyImag/baseEntropyMagni_prun/ReInitialize/model/'+ name_model)    	   	    #set your own path to save the finetuned model    

acc_curve.append(final_testacc)


wandb.finish()

