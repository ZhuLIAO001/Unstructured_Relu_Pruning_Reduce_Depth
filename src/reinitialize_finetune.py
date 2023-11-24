# We reinitialize EGP modeland finetune it. To see can we successfully train from scratch a shallower model, without resorting to an iterative strategy?

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np
from tqdm import tqdm
import copy
import random
import os
from torch.utils.data import DataLoader
import argparse


class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)                        
		else:
			self.hook = module.register_backward_hook(self.hook_fn)													 
	def hook_fn(self, module, input, output):
		self.output = input[0]
	def close(self):
		self.hook.remove()


def train(model, epoch, args, train_loader):
	print('\nEpoch : %d'%epoch)    
	model.train()
	
	running_loss=0
	correct=0
	total=0    
	# loss_fn=torch.nn.CrossEntropyLoss()
	for data in tqdm(train_loader):       
		inputs,labels=data[0].to(args.device),data[1].to(args.device)        
		outputs=model(inputs)       
		# loss=loss_fn(outputs,labels)   
		loss=args.loss_fn(outputs,labels)    
		args.optimizer.zero_grad()
		loss.backward()
		args.optimizer.step()     
		running_loss += loss.item()        
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
		
	train_loss=running_loss/len(train_loader)
	accu=100.*correct/total
		
	print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
	return(accu, train_loss)


def test(model, args, test_loader):
	model.eval()
	
	running_loss=0
	correct=0
	total=0    
	with torch.no_grad():
		for data in tqdm(test_loader):
			images,labels=data[0].to(args.device),data[1].to(args.device)  
			outputs=model(images)
			loss=args.loss_fn(outputs,labels)    
			running_loss+=loss.item()     
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()   
	test_loss=running_loss/len(test_loader)
	accu=100.*correct/total
	

	print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
	return(accu, test_loss)



def main():
	parser = argparse.ArgumentParser(description='Entropy Guided Prunning')
	parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
						help='input batch size for training (default: 128)')
	parser.add_argument('--epochs', type=int, default=160, metavar='N',
						help='number of epochs to train (default: 160)')
	parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
						help='learning rate (default: 0.1)')
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--gamma', type=float, default=0.1)
	parser.add_argument('--milestones', type=list, default=[80,120])
	parser.add_argument('--momentum-sgd', type=float, default=0.9, metavar='M',
						help='Momentum')
	parser.add_argument('--dev', default="cpu")
	parser.add_argument('--seed', type=int, default=43, metavar='S',
						help='random seed (default: 43)')
	parser.add_argument('--datapath', default='data/')
	parser.add_argument('--model', default='Resnet18')
	parser.add_argument('--dataset', default='cifar-10')
	args = parser.parse_args()

	# set random seeds, make results reproduceable
	torch.manual_seed(args.seed)
	os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.use_deterministic_algorithms(True)

	args.device = torch.device(args.dev)
	if args.dev != "cpu":
		torch.cuda.set_device(args.device)

	if args.dataset == 'cifar-10' :        
		# Image preprocessing modules
		transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32,4),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		train_dataset = torchvision.datasets.CIFAR10(root=args.datapath+"cifar-10/",
													train=True, 
													transform=transform,
													download=True)
		test_dataset = torchvision.datasets.CIFAR10(root=args.datapath+"cifar-10/",
													train=False, 
													transform=transforms.Compose([transforms.ToTensor(),
																				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
													download=True)
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
												batch_size=args.batch_size, 
												shuffle=True,
												num_workers = 4)
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
												batch_size=args.batch_size, 
												shuffle=False,
												num_workers = 4)
		if args.model == 'Resnet18':
			model= torch.load('models/Resnet18_cifar-10/sparsity_0.99609375').to(args.device) 
		elif args.model == 'Swin-T':
			model= torch.load('models/Swin-T_cifar-10/sparsity_0.9375').to(args.device) 
		else:
			print('ERROR: Model-Dataset pair not recognized! Only accept Swin-T, Resnet18 for cifar-10, Tiny-Inet.')
			return
	elif args.dataset ==  'Tiny-Inet':
		transform_train = transforms.Compose([
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		transform_test = transforms.Compose([
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		train_dataset = torchvision.datasets.ImageFolder(args.datapath+"tiny-imagenet-200/train/", transform=transform_train)
		test_dataset = torchvision.datasets.ImageFolder(args.datapath+"tiny-imagenet-200/val/", transform=transform_test)
		train_loader = DataLoader(train_dataset,
								batch_size=args.batch_size, 
								shuffle=True, 
								num_workers=4)		
		test_loader = DataLoader(test_dataset,
								batch_size=args.batch_size, 
								shuffle=False, 
								num_workers=4)
		if args.model == 'Resnet18':
			from torchvision.models.resnet import BasicBlock
			from torchvision.models.resnet import ResNet
			class BasicBlock_new(BasicBlock):
				def __init__(self, *args, **kwargs):
					super(BasicBlock_new, self).__init__(*args, **kwargs)
					self.relu1 = nn.ReLU(inplace=True)
					self.relu2 = nn.ReLU(inplace=True)
					delattr(self, 'relu')
				def forward(self, x):
					identity = x
					out = self.conv1(x)
					out = self.bn1(out)
					out = self.relu1(out)
					out = self.conv2(out)
					out = self.bn2(out)
					if self.downsample is not None:
						identity = self.downsample(x)
					out += identity
					out = self.relu2(out)
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
			model= torch.load('models/Resnet18_Tiny-Inet/sparsity_0.9375').to(args.device) 
		elif args.model == 'Swin-T':
			model= torch.load('models/Swin-T_Tiny-Inet/sparsity_0.5').to(args.device) 
		else:
			print('ERROR: Model-Dataset pair not recognized! Only accept Swin-T, Resnet18 for cifar-10, Tiny-Inet.')
			return
	else:
		print('ERROR: Model-Dataset pair not recognized! Only accept Swin-T, Resnet18 for cifar-10, Tiny-Inet.')
		return

	hooks = {}
	if args.model == 'Resnet18':
		for name, module in model.named_modules():
			if type(module) == torch.nn.ReLU:
				hooks[name] = Hook(module)
	elif args.model == 'Swin-T':
		for name, module in model.named_modules():
			if type(module) == torch.nn.GELU:
				hooks[name] = Hook(module)

	# Reinilize the shallower model
	if args.model == 'Resnet18':
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
	elif args.model == 'Swin-T':
		for name, module in list(model.named_modules()):
			if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
				if torch.numel(torch.abs(module.weight)[module.weight != 0])==0:
					for param in module.parameters():
						param.requires_grad = False
				else:
					torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=1.0, a=- 0.01, b=0.01)
			elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)


	args.loss_fn = torch.nn.CrossEntropyLoss().to(args.device)
	args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(args.optimizer, milestones=args.milestones, gamma=args.gamma)
	for epoch in range(1, args.epochs+1):
		train_acc, train_loss = train(model, epoch,  args, train_loader)
		test_acc, test_loss = test(model,  args, test_loader)
		scheduler.step()
	torch.save(model, 'Reiitialized_models/'+ args.model+'_' + args.dataset)                   #set your own path to save model
	


if __name__ == '__main__':
	main()









