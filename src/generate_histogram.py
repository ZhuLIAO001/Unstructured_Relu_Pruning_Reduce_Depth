# Calculate the entropy of relu or identity layers' input, and generate the histogram

import numpy as np
import torch
import torchvision
import torch.nn.utils.prune
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import random


#Hook on relu layer
class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)
		else:
			self.hook = module.register_backward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		self.output = torch.heaviside(input[0] ,torch.tensor([0],dtype=torch.float32).to(device))
	def close(self):
		self.hook.remove()


#Hook on identity layer
class Hook_Identity():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)
		else:
			self.hook = module.register_backward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		self.output =   torch.where(input[0]==0, 0, torch.ones_like(input[0])).to(device)
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

def replace_relu_inplace(model):
    for name, module in model.named_children():
        if type(module) == torch.nn.ReLU:
            setattr(model, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu_inplace(module) 

def main():
	parser = argparse.ArgumentParser(description='Generate Histogram')
	parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
						help='input batch size for training (default: 128)')
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
		num_classes = 10
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
			model_path =  'models/Resnet18_cifar-10'
		elif args.model == 'Swin-T':
			model_path =  'models/Swin-T_cifar-10'
		else:
			print('ERROR: Model-Dataset pair not recognized! Only accept Swin-T, Resnet18 for cifar-10, Tiny-Inet.')
			return
	elif args.dataset ==  'Tiny-Inet':
		num_classes = 200
		transform_train = transforms.Compose([
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		transform_test = transforms.Compose([
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		train_dataset = torchvision.datasets.ImageFolder(args.datapath+"tiny-imagenet-200/train/", transform=transform_train)
		test_dataset = torchvision.datasets.ImageFolder(args.datapath+"tiny-imagenet-200/val/", transform=transform_test)
		train_loader = torch.utils.data.DataLoader(train_dataset,
								batch_size=args.batch_size, 
								shuffle=True, 
								num_workers=4)		
		test_loader = torch.utils.data.DataLoader(test_dataset,
								batch_size=args.batch_size, 
								shuffle=False, 
								num_workers=4)
		if args.model == 'Resnet18':
			model_path =  'models/Resnet18_Tiny-Inet'
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
		elif args.model == 'Swin-T':
			model_path =  'models/Swin-T_Tiny-Inet'
		else:
			print('ERROR: Model-Dataset pair not recognized! Only accept Swin-T, Resnet18 for cifar-10, Tiny-Inet.')
			return
	else:
		print('ERROR: Model-Dataset pair not recognized! Only accept Swin-T, Resnet18 for cifar-10, Tiny-Inet.')
		return


	g = os.walk(r'model_path')                    
	for root, dir_list, file_list in g:
		for file_name in file_list:
			print('file name:',file_name)

			num_filter_layer = {}
			layer_name = []
			layer_act_0_per = []
			layer_act_1_per = []
			layer_act_0_num = []
			layer_act_1_num = []
			layer_act_mix_num = []
			model= torch.load('model_path'+'/'+'file_name').to(args.device)            #Your path, load the models
			model.eval()

			hooks = {}
			if args.model == 'Resnet18':
				replace_relu_inplace(model)
				for name, module in model.named_modules():
					replace_relu_inplace(model)
					if type(module) == torch.nn.ReLU:
						hooks[name] = Hook(module, backward=False)
					if type(module) == torch.nn.Identity:
						hooks[name] = Hook_Identity(module, backward=False)
			elif args.model == 'Swin-T':
				for name, module in model.named_modules():
					if type(module) == torch.nn.GELU:
						hooks[name] = Hook(module, backward=False)
					if type(module) == torch.nn.Identity:
						hooks[name] = Hook_Identity(module, backward=False)

			with torch.no_grad():			
				joint_activation_logger = {}
				entorpy0_position = {} 																												
				for key in hooks.keys():                               
					joint_activation_logger[key] = {}
					entorpy0_position[key] = {} 


				for key in entorpy0_position.keys():
					layer_name.append(key)
					entorpy0_position[key][0] = {}
					entorpy0_position[key][1] = {}

				with torch.no_grad():
					for data in tqdm(train_loader):
						images,labels=data[0].to(args.device),data[1].to(args.device) 
						outputs= model(images)             	
						for key in joint_activation_logger.keys():
							p_one = torch.mean(hooks[key].output,dim=0)
							while len(p_one.shape) > 1:       			
								p_one = torch.mean(p_one,dim=0)
							for j in range(p_one.shape[0]):
								num_filter_layer[key] = p_one.shape[0]
								joint_activation_logger[key][j] = 0
						break;         
					for data in tqdm(train_loader):
						images,labels=data[0].to(args.device),data[1].to(args.device)  
						outputs=model(images)             	
						for key in joint_activation_logger.keys():
							p_one = torch.mean(hooks[key].output,dim=0)
							print(p_one.shape)                 
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

		test_acc, test_loss = test(model)



		fig = plt.figure(figsize= (20,10),dpi = 500)        

		p1 = plt.bar(layer_name, layer_act_0_num, width=0.4, color='darkorange', label='Nonlinear')    
		p2 = plt.bar(layer_name, layer_act_1_num, width=0.4, color='indianred', bottom=layer_act_0_num, label='linear')
		p3 = plt.bar(layer_name, layer_act_mix_num, width=0.4, color='royalblue', bottom=list(np.add(layer_act_0_num,layer_act_1_num)), label='mix')        
		plt.title('number of neurons always linear/nonlinear with acc:'+ str(test_acc))                                                  
		plt.legend() 

		plt.legend([r'OFF ($\mathcal{H}(l,i|\Xi)=0$)', r'ON ($\mathcal{H}(l,i|\Xi)=0$)', r'$\mathcal{H}(l,i|\Xi) \neq 0$'], fontsize=25) 
		plt.grid(True, linestyle='--', alpha=0.5)
		plt.xlabel("Layer index", fontsize=20)
		plt.ylabel("Neurons", fontsize=20)
		# plt.xticks(size=8)
		plt.yticks(size=20)

		plt.bar_label(p1, label_type='center')
		plt.bar_label(p2, label_type='center')
		plt.bar_label(p3, label_type='center')



		plt.savefig('histograms/'+ args.model+'_' + args.dataset+'/'+ file_name + '.pdf')                    # set your own path to save the histogram




if __name__ == '__main__':
	main()