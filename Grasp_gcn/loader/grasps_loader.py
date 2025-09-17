#import logging
#import torch
#import torch.utils.data
#import numpy as np
#import pandas as pd

import os
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
#from transforms.tograph import ToGraph
import numpy as np

#import sys
#sys.path.append('/home/nadia/env-AIST/GIT/AIST/env/transforms')  # Adjust the path as necessary
from transforms.tograph import ToGraph


class GraspsDataset(Dataset):
    def __init__(self, root, csv_file, transform=ToGraph(), pre_transform=None):
        self.csv_file = csv_file
        super().__init__(root, transform, pre_transform)
        self.data = pd.read_csv(os.path.join(root, csv_file))

    @property
    def raw_file_names(self):
        #return [self.csv_file]
        return ['grasps_sample_train.csv']

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.data))]

    def download(self):
        # Download data if not already downloaded
        pass

    def process(self):
        for idx, row in self.data.iterrows():
            sample = {
                'object': row[0],
                'grasp_type': row[1],
                'WRIST': row[2],
                'THUMB_CMC': row[3],
                'THUMB_MCP': row[4],
                'THUMB_IP': row[5],
                'THUMB_TIP': row[6],
                'INDEX_FINGER_MCP': row[7],
                'INDEX_FINGER_PIP': row[8],
                'INDEX_FINGER_DIP': row[9],
                'INDEX_FINGER_TIP': row[10],
                'MIDDLE_FINGER_MCP': row[11],
                'MIDDLE_FINGER_PIP': row[12],
                'MIDDLE_FINGER_DIP': row[13],
                'MIDDLE_FINGER_TIP': row[14],
                'RING_FINGER_MCP': row[15],
                'RING_FINGER_PIP': row[16],
                'RING_FINGER_DIP': row[17],
                'RING_FINGER_TIP': row[18],
                'PINKY_MCP': row[19],
                'PINKY_PIP': row[20],
                'PINKY_DIP': row[21],
                'PINKY_TIP': row[22]
                # Add all joint data processing here as in your current setup
            }
            data = self.transform(sample) if self.transform else sample
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


##############################################################################old version

#log = logging.getLogger(__name__)

#class GraspsDataset(torch.utils.data.Dataset):
#	"""Grasping Types Dataset"""

	#def __init__(self, csvFile, transform=None):
#	def __init__(self, train_csvs, transform=None):
#		"""
#		Args:
#			csvFile (string): Path to the CSV file with annotations.
#		"""
#		#self.m_csv_file = csvFile
#		self.m_csv_file = train_csvs
#		self.m_grasps = pd.read_csv("data/" + self.m_csv_file)
#		self.m_transform = transform

#	def __len__(self):
#		
#		return len(self.m_grasps)

#defining the graph info structure.

#	def __getitem__(self, idx):

#		sample_ = self.m_grasps.iloc[idx]

#		object_ = sample_.iloc[0]
#		grasp_type_ = sample_.iloc[1]
#		WRIST_= np.copy(sample_.iloc[2:5]).astype(np.float32, copy=False)
#		THUMB_CMC_ = np.copy(sample_.iloc[5:8]).astype(np.float32, copy=False)
#		THUMB_MCP_ = np.copy(sample_.iloc[8:11]).astype(np.float32, copy=False)
#		THUMB_IP_ = np.copy(sample_.iloc[11:14]).astype(np.float32, copy=False)
#		THUMB_TIP_ = np.copy(sample_.iloc[14:17]).astype(np.float32, copy=False)
#		INDEX_FINGER_MCP_ = np.copy(sample_.iloc[17:20]).astype(np.float32, copy=False)
#		INDEX_FINGER_PIP_ = np.copy(sample_.iloc[20:23]).astype(np.float32, copy=False)
#		INDEX_FINGER_DIP_ = np.copy(sample_.iloc[23:26]).astype(np.float32, copy=False)
#		INDEX_FINGER_TIP_ = np.copy(sample_.iloc[26:29]).astype(np.float32, copy=False)
#		MIDDLE_FINGER_MCP_ = np.copy(sample_.iloc[29:32]).astype(np.float32, copy=False)
#		MIDDLE_FINGER_PIP_ = np.copy(sample_.iloc[32:35]).astype(np.float32, copy=False)
#		MIDDLE_FINGER_DIP_ = np.copy(sample_.iloc[35:38]).astype(np.float32, copy=False)
#		MIDDLE_FINGER_TIP_ = np.copy(sample_.iloc[38:41]).astype(np.float32, copy=False)
#		RING_FINGER_MCP_ =  np.copy(sample_.iloc[41:44]).astype(np.float32, copy=False)
#		RING_FINGER_PIP_ = np.copy(sample_.iloc[44:47]).astype(np.float32, copy=False)
#		RING_FINGER_DIP_ = np.copy(sample_.iloc[47:50]).astype(np.float32, copy=False)
#		RING_FINGER_TIP_ = np.copy(sample_.iloc[50:53]).astype(np.float32, copy=False)
#		PINKY_MCP_ = np.copy(sample_.iloc[53:56]).astype(np.float32, copy=False)
#		PINKY_PIP_ = np.copy(sample_.iloc[56:59]).astype(np.float32, copy=False)
#		PINKY_DIP_ = np.copy(sample_.iloc[59:62]).astype(np.float32, copy=False)
#		PINKY_TIP_ = np.copy(sample_.iloc[62:65]).astype(np.float32, copy=False)

#		sample_ = {'object': object_,
#							'grasp_type': grasp_type_,
#							'WRIST': WRIST_,
#							'THUMB_CMC': THUMB_CMC_,
#							'THUMB_MCP': THUMB_MCP_,
#							'THUMB_IP': THUMB_IP_,
#							'THUMB_TIP': THUMB_TIP_,
#							'INDEX_FINGER_MCP': INDEX_FINGER_MCP_,
#							'INDEX_FINGER_PIP': INDEX_FINGER_PIP_,
#							'INDEX_FINGER_DIP': INDEX_FINGER_DIP_,
#							'INDEX_FINGER_TIP': INDEX_FINGER_TIP_,
#							'MIDDLE_FINGER_MCP': MIDDLE_FINGER_MCP_,
#							'MIDDLE_FINGER_PIP': MIDDLE_FINGER_PIP_,
#							'MIDDLE_FINGER_DIP': MIDDLE_FINGER_DIP_,
#							'MIDDLE_FINGER_TIP': MIDDLE_FINGER_TIP_,
#							'RING_FINGER_MCP': RING_FINGER_MCP_,
#							'RING_FINGER_PIP': RING_FINGER_PIP,
#							'RING_FINGER_DIP': RING_FINGER_DIP,
#							'RING_FINGER_TIP': RING_FINGER_TIP,
#							'PINKY_MCP': PINKY_MCP_,
#							'PINKY_PIP': PINKY_PIP_,
#							'PINKY_DIP': PINKY_DIP_,
#							'PINKY_TIP': PINKY_PIP_}
#		if self.m_transform:
#			sample_ = self.m_transform(sample_)

#		return sample_

#	def __repr__(self):

#		return "Dataset loader for Grasps with {0} entries in {1}".format(
#			self.__len__(),
#			self.m_csv_file
#		)
