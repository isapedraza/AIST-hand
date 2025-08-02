import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold

import argparse
import datetime
import logging
import sys
import time
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
#from torch_geometric.data import DataLoader, this is not used anymore
from torch_geometric.loader import DataLoader


import loader.grasps_loader
import dataset.grasps
import network.utils
import transforms.tograph
import utils.evaluation
import utils.plotaccuracies
import utils.plotlosses
import matplotlib.pyplot as plt

#import torch nn
import torch.nn as nn

print('--------------------------------')
print(f'import libraries okay')


############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/testJune17')
###################################################

# Hyper-parameters 
lr = 0.0001 #Learning Rate
normalize = True #Normalize dataset
#network_type = "GCN_test" #class selected on network and utils
network_type = "GCN_8_8_16_16_32" #This is the network model to train
weight_decay=5e-4

seed=1

BATCH_SIZE=1
visualize_batch=False #careful with sintax! false is non existent
num_epochs=10 #Training Epochs

print('--------------------------------')
print(f'batch size: {BATCH_SIZE}')
print('--------------------------------')
print(f'num epochs: {num_epochs}')


NUM_WORKERS = 1 #4 
#This parameter determines the number of subprocesses to use for data loading. Each worker independently loads a batch of data in parallel with others. 

print('--------------------------------')
print(f'LOGGER and num workers: {NUM_WORKERS}')

#print(sample_.keys()) 


from transforms.tograph import ToGraph

# Initialize dataset
train_csvs = "/home/nadia/env-AIST/data/grasps_sample_train.csv"

from dataset.grasps import GraspsClass

datasetTrain = dataset.grasps.GraspsClass(root='data/', split=None, transform=ToGraph(), csvs=train_csvs, normalize=normalize)
    
data_size = len(datasetTrain)

print("Data size:", data_size)

test = datasetTrain[0] #not to confuse with sample_ def
print(test.keys())

# Create an instance of ToGraph
#to_graph_transform = ToGraph()

# Transform each sample in the dataset
#transformed_data_list = []
#for i in range(len(datasetTrain)):
#    sample = datasetTrain[i]
    # Print the keys of the sample to debug
#    print(f'Sample {i} keys: {sample.keys()}')
#    transformed_sample = to_graph_transform(sample)
#    transformed_data_list.append(transformed_sample)


#print('--------------------------------')
#print(f'Transformed data length: {len(transformed_data_list)}')
#print(f'Example of transformed data (10): {transformed_data_list[10]}')

# Transform the sample_data to a torch_geometric Data object
#transformed_data = to_graph_transform(datasetTrain)


def reset_weights(m):
  '''
    resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters() 


print('--------------------------------')
print(f'reset weight')


## Select CUDA device
device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('--------------------------------')
print(f'device selection:')
print('--------------------------------')
print(f'{device_}')
    

#BUILD MODEL
from network.utils import get_network

model_ = get_network(network_type, datasetTrain.num_features, datasetTrain.num_classes).to(device_)

def train(model_, datasetTrain, writer):
    print('--------------------------------')
    print(f'test inside def train')
    
    train_loader_ = DataLoader(datasetTrain, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) 
    

    print('--------------------------------')
    print(f'print loader pass')
 
    torch.cuda.manual_seed(seed)

    print(f'train loader test')
    print(len(train_loader_))        

    model_.apply(reset_weights)


    ## Optimizer
   
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=lr, weight_decay=weight_decay)

## Log accuracies, learning rate
    epochs_ = []
    best_test_acc_ = 0.0
    train_accuracies_ = []
    test_accuracies_ = []
    train_losses_ = []

    time_start_ = timer()
    
    
    print('--------------------------------')
    print(f'def train test')
        
    
    #for epoch in range(200):
    for epoch in range(0, num_epochs): 
        model_.train()
        loss_all = 0

        i = 1
        for batch in train_loader_:
            optimizer_.zero_grad()
            pred = model_(batch)
            label = batch.y
            loss = F.nll_loss(pred, label)
            loss.backward()
            optimizer_.step()
            loss_all += loss.item() * batch.num_graphs
        loss_all /= len(train_loader_.dataset)
        writer.add_scalar("loss", loss_all, epoch)

        # Log train loss
        train_losses_.append(loss_all)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_all:.4f}')

    return model_

trained_model = train(model_, datasetTrain, writer)

    ######################################################################################
     # Process is complete.
print('Training process has finished. Saving trained model.')
