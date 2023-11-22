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
import os
from torch.utils.data import DataLoader
import argparse

## Choose device
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0")



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

    args = parser.parse_args()
    args.device = torch.device(args.dev)
    if args.dev != "cpu":
        torch.cuda.set_device(args.device)
    args.loss_fn = torch.nn.CrossEntropyLoss().to(args.device)
    args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)


if __name__ == '__main__':
    main()