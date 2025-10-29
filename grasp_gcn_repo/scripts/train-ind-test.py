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
from torch_geometric.data import DataLoader

import loader.grasps_loader
from grasp_gcn.dataset import grasps
#import network.utils
from grasp_gcn.transforms.tograph import ToGraph
#import transforms.addnoise
from grasp_gcn.utils import evaluation
from grasp_gcn.utils import plotaccuracies
#import utils.plotcontour
#import utils.plotgraph
from grasp_gcn.utils import plotlosses


#import torch
import torch.nn as nn
import torch.nn.functional as F


import os



# Print
   
print('--------------------------------')
print(f'import libraries okay')

#lr = 0.0001
weight_decay=5e-4

#the args variable typically becomes a dictionary or another data structure that holds the configuration parameters you need for your training and evaluation processes. You would manually populate this data structure with the necessary values before calling the train function.

#so we define it here
args = {
    'train_csvs': ['/home/nadia/env-AIST/data/grasps_sample_train.csv'],
    'test_csvs': ['/home/nadia/env-AIST/data/grasps_sample_test.csv'],
    'log_path': 'logs',
    'ckpt_path': 'ckpts',
    'save_ckpt': True,
    'normalize': True,
    'batch_size': 1,
    'network': 'GCN_test',
    'lr': 0.0001,
    'epochs': 32,
    'visualize_batch': False,
    # Add other parameters as needed
}

print('--------------------------------')
print(f'defining arg dictionary okay')

# Ensure log directory exists
log_dir = args.get("log_path")
os.makedirs(log_dir, exist_ok=True)


#print('test args: {args.get("train_csvs")}')

print(f'test args: {args.get("train_csvs")}')



BATCH_SIZE=1
visualize_batch=False #careful with sintax! false is non existent
#epochs=32 #for initial test #512 gcntactile
num_epochs=10

print('--------------------------------')
print(f'batch size: {BATCH_SIZE}')

print(f'num epochs: {num_epochs}')
#num_workers=1

log = logging.getLogger(__name__)

NUM_WORKERS = 1 #4

print('--------------------------------')
print(f'LOGGER and num workers: {NUM_WORKERS}')
#def visualize_batch(batch):

#    log.info(batch)
#    log.info("Batch size {}".format(batch.num_graphs))


def set_seed(seed=1):
    np.random.seed(seed)
#    torch.manual_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
print('--------------------------------')
print(f'fix seed')

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

def traintest(args, experimentStr, datasetTrain, datasetTest):

    log.info("Training and testing...")

    train_loader_ = DataLoader(datasetTrain, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS) #
    test_loader_ = DataLoader(datasetTest, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #This parameter determines the number of subprocesses to use for data loading. Each worker independently loads a batch of data in parallel with others. 

    print('--------------------------------')
    print(f'print loader pass')

    ## Select CUDA device
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(device_)
    log.info(torch.cuda.get_device_name(0))
 
    print('--------------------------------')
    print(f'device selection:')
    print('--------------------------------')
    print(f'{device_}')
    
    torch.cuda.manual_seed(seed)
        
# Init the neural network/build model
#  model_ = GCN_8_8_16_16_32()
    model_ = network.utils.get_network(network, datasetTrain.data.num_features, datasetTrain.data.num_classes).to(device_)
    model_.apply(reset_weights)

    log.info(model_)
  #return model_

    ## Optimizer
   # optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.get("lr"), weight_decay=weight_decay)
    log.info(optimizer_)
## Log accuracies, learning rate, and loss
    epochs_ = []
    best_test_acc_ = 0.0
    train_accuracies_ = []
    test_accuracies_ = []
    train_losses_ = []

    time_start_ = timer()
    
    
print('--------------------------------')
print(f'def traintest')
        
    
    #for epoch in range(args.epochs):
for epoch in range(0, num_epochs):
  log.info("Training epoch {0} out of {1}".format(epoch, num_epochs))
      
  print(f'Starting epoch {epoch+1}')
      
  model_.train()
  loss_all = 0

  i = 1
  for batch in train_loader_:

      # Batch Visualization
      if (args.visualize_batch):
          log.info("Training batch {0} of {1}".format(i, len(dataset)/args.batch_size))
          visualize_batch(batch)

          batch = batch.to(device_)
          optimizer_.zero_grad() #zero the gradients
          output_ = model_(batch)
          loss_ = F.nll_loss(output_, batch.y)
          loss_.backward()
          loss_all += batch.y.size(0) * loss_.item()
          optimizer_.step()

          i+=1

        # Log train loss
      train_losses_.append(loss_all)
      log.info("Training loss {0}".format(loss_all))

        # Get train accuracy
      model_.eval()
      correct_ = 0

      for batch in train_loader_:

          batch = batch.to(device_)
          pred_ = model_(batch).max(1)[1]
          correct_ += pred_.eq(batch.y).sum().item()

      correct_ /= len(datasetTrain)

        # Log train accuracy
      train_accuracies_.append(correct_)
      log.info("Training accuracy {0}".format(correct_))

        # Get test accuracy
    #  model_.eval()
    #  correct_ = 0

    #  for batch in test_loader_:

    #      batch = batch.to(device_)
    #      pred_ = model_(batch).max(1)[1]
    #      correct_ += pred_.eq(batch.y).sum().item()

    #  correct_ /= len(datasetTest)

        # Log test accuracy
   #   test_accuracies_.append(correct_)
   #   log.info("Test accuracy {0}".format(correct_))

    # Checkpoint model
    #  if correct_ > best_test_acc_ and args.save_ckpt:

    #      log.info("BEST ACCURACY SO FAR, checkpoint model...")

    #      best_test_acc_ = correct_

      state_ = {'epoch': epoch+1,
                 'model_state': model_.state_dict(),
                 'optimizer_state': optimizer_.state_dict(),}
      torch.save(state_, (args.ckpt_path + "/" + experimentStr + "_{0}.pkl").format(epoch))

      epochs_.append(epoch)

    time_end_ = timer()
    log.info("Training took {0} seconds".format(time_end_ - time_start_))

    #utils.plotaccuracies.plot_accuracies(epochs_, [train_accuracies_, test_accuracies_], ["Train Accuracy", "Test Accuracy"])
    utils.plotlosses.plot_losses(epochs_, [train_losses_], ["Train Loss"])

def train(dataset, experimentStr):

    grasps_dataset_train_ = dataset.grasps.GraspsClass(root='data/', split=None, csvs=args.get(train_csvs), normalize=args.normalize)
    grasps_dataset_test_ = dataset.grasps.GraspsClass(root='data/', split=None, csvs=args.get(test_csvs), normalize=args.normalize)

    log.info(grasps_dataset_train_)
    log.info(grasps_dataset_test_)
    traintest(args, experimentStr, grasps_dataset_train_, grasps_dataset_test_)

if __name__ == "__main__":

    # Experiment name (and log filename) follows the format network-normalization-graph_k-datetime
   # experiment_str_ = "traintest-{0}-{1}-{2}-{3}-{4}-{5}".format(
    experiment_str_ = "traintest-{0}-{1}-{2}-{3}".format(
                        ''.join(args.get("train_csvs")), 
                       # ''.join(args.get(test_csvs)),
                        args.get("network"),
                        args.get("normalize"),
                       # args_.graph_k,
                        datetime.datetime.now().strftime('%b%d_%H-%M-%S'))

    # Add file handler to logging system to simultaneously log information to console and file
    log_formatter_ = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    #file_handler_ = logging.FileHandler("{0}/{1}.log".format(args.get("log_path"), experiment_str_))
    file_handler_ = logging.FileHandler(os.path.join(args.get("log_path"), f"{experiment_str_}.log"))
    file_handler_.setFormatter(log_formatter_)
    log.addHandler(file_handler_)

    train(args, experiment_str_)
    
     # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Testing starts')
    
    # Saving the model
    save_path = f'./model.pth'
    torch.save(model_.state_dict(), save_path)

   # return model_
