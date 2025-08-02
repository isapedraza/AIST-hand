import os
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset

#import sys
#sys.path.append('/home/nadia/env-AIST/GIT/AIST/env-AIST') 

import transforms.tograph
#from transforms import tograph
#from transforms.tograph import ToGraph
#from env.transforms import tograph

#import os

#print(os.getcwd())

k = 0

log = logging.getLogger(__name__)

class GraspsClass(InMemoryDataset):

  def __init__(self, root,  split="train", normalize=True, csvs=None, transform=None, pre_transform=None):

    self.split = split
    self.csvs = csvs
    #self.k = k # I am defining manually the connections so no k definition for my method.
    self.normalize = normalize
    self.mins = []
    self.maxs = []

    super(GraspsClass, self).__init__(root, transform, pre_transform)

    self.data, self.slices = torch.load(self.processed_paths[0])

    # Compute class weights for sampling
    self.class_weights = np.zeros(3) #number of grasp classes
    for i in range(len(self.data['y'])):
      self.class_weights[self.data['y'][i]] += 1
    self.class_weights /= len(self.data['y'])

  @property
  def raw_file_names(self):
    if (self.split == "train"):
      return ['grasps_sample_train.csv']
    elif (self.split == "test"):
      return ['grasps_sample_test.csv']
    elif (self.split == None):
      return self.csvs

  @property
  def processed_file_names(self):
    if (self.split == "train"):
      return ["grasps_{0}.pt".format(self.k)]
    elif (self.split == "test"):
      return ["grasps_test_{0}.pt".format(self.k)]
    elif (self.split == None):
      return ['grasps_' + ''.join(self.csvs) + '.pt']

  def process(self):

    #transform_tograph_ = transforms.tograph.ToGraph(self.k)
    transform_tograph_ = transforms.tograph.ToGraph()
    
    data_list_ = []

    for f in range(len(self.raw_paths)):

      log.info("Reading CSV file {0}".format(self.raw_paths[f]))

      grasps_ = pd.read_csv(self.raw_paths[f])
      
      print(f"Data from {self.raw_paths[f]} loaded successfully. First few rows:")
      print(grasps_.head())

      for i in range(len(grasps_)):

        sample_ = self._sample_from_csv(grasps_, i)
        sample_ = transform_tograph_(sample_)

        if self.pre_transform is not None:
          sample_ = self.pre_transform(sample_)

        data_list_.append(sample_)

   # if self.normalize: # Feature scaling
   #   raw_dataset_np_ = np.array([sample.x.numpy() for sample in data_list_])

    #  self.mins = np.min(raw_dataset_np_, axis=(0, 1))
    #  self.maxs = np.max(raw_dataset_np_, axis=(0, 1))

     # for i in range(len(data_list_)):
      #  data_list_[i].x = torch.from_numpy((data_list_[i].x.numpy() - self.mins) / (self.maxs - self.mins))

    if self.normalize:  # Standardization
      #raw_dataset_np_ = np.vstack([sample.x.numpy() for sample in data_list_]) #sample can be any variable name
      raw_dataset_np_ = np.vstack([sample['x'].numpy() for sample in data_list_])
      mean = np.mean(raw_dataset_np_, axis=0)
      std = np.std(raw_dataset_np_, axis=0)
      print(f"Mean: {mean}, Std: {std}")

      for i in range(len(data_list_)):
        data_list_[i].x = torch.from_numpy((data_list_[i].x.numpy() - mean) / std)

    data_ = self.collate(data_list_)

# Ensure the directory exists
    os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
    
    torch.save(data_, self.processed_paths[0])

  #def _sample_from_csv(self, grasps, idx):
  #test
  def _sample_from_csv(self, grasps, idx):
  ####

    sample_ = self.m_grasps.iloc[idx]
    #sample_ = grasps.iloc[idx]
                
    object_ = sample_.iloc[0]
    grasp_type_ = sample_.iloc[1]
    WRIST_= np.copy(sample_.iloc[2:5]).astype(float32, copy=False)
    THUMB_CMC_ = np.copy(sample_.iloc[5:8]).astype(float32, copy=False)
    THUMB_MCP_ = np.copy(sample_.iloc[8:11]).astype(float32, copy=False)
    THUMB_IP_ = np.copy(sample_.iloc[11:14]).astype(float32, copy=False)
    THUMB_TIP_ = np.copy(sample_.iloc[14:17]).astype(float32, copy=False)
    INDEX_FINGER_MCP_ = np.copy(sample_.iloc[17:20]).astype(float32, copy=False)
    INDEX_FINGER_PIP_ = np.copy(sample_.iloc[20:23]).astype(float32, copy=False)
    INDEX_FINGER_DIP_ = np.copy(sample_.iloc[23:26]).astype(float32, copy=False)
    INDEX_FINGER_TIP_ = np.copy(sample_.iloc[26:29]).astype(float32, copy=False)
    MIDDLE_FINGER_MCP_ = np.copy(sample_.iloc[29:32]).astype(float32, copy=False)
    MIDDLE_FINGER_PIP_ = np.copy(sample_.iloc[32:35]).astype(float32, copy=False)
    MIDDLE_FINGER_DIP_ = np.copy(sample_.iloc[35:38]).astype(float32, copy=False)
    MIDDLE_FINGER_TIP_ = np.copy(sample_.iloc[38:41]).astype(float32, copy=False)
    RING_FINGER_MCP_ =  np.copy(sample_.iloc[41:44]).astype(float32, copy=False)
    RING_FINGER_PIP_ = np.copy(sample_.iloc[44:47]).astype(float32, copy=False)
    RING_FINGER_DIP_ = np.copy(sample_.iloc[47:50]).astype(float32, copy=False)
    RING_FINGER_TIP_ = np.copy(sample_.iloc[50:53]).astype(float32, copy=False)
    PINKY_MCP_ = np.copy(sample_.iloc[53:56]).astype(float32, copy=False)
    PINKY_PIP_ = np.copy(sample_.iloc[56:59]).astype(float32, copy=False)
    PINKY_DIP_ = np.copy(sample_.iloc[59:62]).astype(float32, copy=False)
    PINKY_TIP_ = np.copy(sample_.iloc[62:65]).astype(float32, copy=False)

#np.int not valid anymore WRIST_= np.copy(sample_.iloc[2:5]).astype(int, copy=False)  # or np.int64 or np.int32 based on your precision needs

    sample_ = {'object': object_,
                                                        'grasp_type': grasp_type_,
                                                        'WRIST': WRIST_,
                                                        'THUMB_CMC': THUMB_CMC_,
                                                        'THUMB_MCP': THUMB_MCP_,
                                                        'THUMB_IP': THUMB_IP_,
                                                        'THUMB_TIP': THUMB_TIP_,
                                                        'INDEX_FINGER_MCP': INDEX_FINGER_MCP_,
                                                        'INDEX_FINGER_PIP': INDEX_FINGER_PIP_,
                                                        'INDEX_FINGER_DIP': INDEX_FINGER_DIP_,
                                                        'INDEX_FINGER_TIP': INDEX_FINGER_TIP_,
                                                        'MIDDLE_FINGER_MCP': MIDDLE_FINGER_MCP_,
                                                        'MIDDLE_FINGER_PIP': MIDDLE_FINGER_PIP_,
                                                        'MIDDLE_FINGER_DIP': MIDDLE_FINGER_DIP_,
                                                        'MIDDLE_FINGER_TIP': MIDDLE_FINGER_TIP_,
                                                        'RING_FINGER_MCP': RING_FINGER_MCP_,
                                                        'RING_FINGER_PIP': RING_FINGER_PIP_,
                                                        'RING_FINGER_DIP': RING_FINGER_DIP_,
                                                        'RING_FINGER_TIP': RING_FINGER_TIP_,
                                                        'PINKY_MCP': PINKY_MCP_,
                                                        'PINKY_PIP': PINKY_PIP_,
                                                        'PINKY_DIP': PINKY_DIP_,
                                                        'PINKY_TIP': PINKY_PIP_}
   # if self.m_transform:
   #     sample_ = self.m_transform(sample_)

    return sample_


