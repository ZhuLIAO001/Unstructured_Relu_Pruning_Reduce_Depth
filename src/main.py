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


def test_entropy(model, hooks, args, train_loader):
	model.eval()

	layers_entropy = {}
	entropy = {}
	for key in hooks.keys():
		entropy[key] = 0
	
	running_loss=0
	correct=0
	total=0    


	with torch.no_grad():

		for data in tqdm(train_loader):
			images,labels=data[0].to(args.device),data[1].to(args.device) 
			outputs=model(images)
			loss=args.loss_fn(outputs,labels)   
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
	parser.add_argument('--fixed_amount_of_pruning', type=float, default=0.5)	
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

	if args.dataset == 'cifar-10' and args.model == 'Resnet18':
		from cifar10_models.resnet import resnet18
		model = resnet18().to(args.device)
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
	
	elif args.dataset == 'Tiny-Inet' and args.model == 'Resnet18':
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
		model = ResNet_new(BasicBlock_new, [2, 2, 2, 2], num_classes=200).to(args.device)
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

	elif args.dataset == 'cifar-10' and args.model == 'Swin-T':
		model = torchvision.models.swin_t(weights = True)
		model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=10).to(args.device)
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

	elif args.dataset == 'Tiny-Inet' and args.model == 'Swin-T':
		model = torchvision.models.swin_t(weights = True)
		model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=1000).to(args.device)
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

	args.loss_fn = torch.nn.CrossEntropyLoss().to(args.device)
	args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(args.optimizer, milestones=args.milestones, gamma=args.gamma)
	for epoch in range(1, args.epochs+1):
		train_acc, train_loss = train(model, epoch,  args, train_loader)
		test_acc, test_loss = test(model,  args, test_loader)
		scheduler.step()
		torch.save(model, 'models/'+ args.model+'_' + args.dataset+'/'+ 'sparsity_0')                   #set your own path to save model

	for i in range(1, 10):
		sparsity = 1-(1-args.fixed_amount_of_pruning)**i
		name_model = 'sparsity_'+str(sparsity)
		test_entropy_acc, test_entropy_loss, layers_entropy = test_entropy(model, hooks, args, train_loader)   #calculate the entropy for each layer
		layers_to_prune=[]
		if args.model == 'Resnet18':
				for key in hooks.keys():
					if key == 'relu':
						layers_to_prune.append('conv1')
						layers_entropy['conv1'] = layers_entropy.pop("relu")
					else:
						name = key.replace('relu', 'conv')
						layers_to_prune.append(name)
						layers_entropy[name] = layers_entropy.pop(key)
		elif args.model == 'Swin-T':
				previous_conv_name = None
				for name, module in model.named_modules():
					if type(module) == torch.nn.Conv2d or type(module) == torch.nn.Linear:
						previous_conv_name = name
					elif name in hooks.keys():
						layers_to_prune.append(previous_conv_name)
						layers_entropy[previous_conv_name] = layers_entropy.pop(name)

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
		total_layers_weight_paras_to_prun = args.fixed_amount_of_pruning * total_layers_weight_paras

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

		args.loss_fn = torch.nn.CrossEntropyLoss().to(args.device)
		args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
		scheduler = torch.optim.lr_scheduler.MultiStepLR(args.optimizer, milestones=args.milestones, gamma=args.gamma)
		for epoch in range(1, args.epochs+1):
			train_acc, train_loss = train(model, epoch,  args, train_loader)
			test_acc, test_loss = test(model,  args, test_loader)
			scheduler.step()
		temp_model = copy.deepcopy(model)
		for name, module in temp_model.named_modules():
			if name in layers_to_prune:
				prune.remove(module,'weight')
		torch.save(temp_model, 'models/'+ args.model+'_' + args.dataset+'/'+ 'sparsity_0')                   #set your own path to save model
	


if __name__ == '__main__':
	main()