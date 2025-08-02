import logging
import torch
from torch_geometric.data import Data
import numpy as np

log = logging.getLogger(__name__)

class ToGraph(object):
    def __init__(self):
     
   # Manual connections
        #self.m_edge_origins = [0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 10, 11, 11, 12, 13, 13, 13, 14, 14, 14, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23]
        #self.m_edge_ends = [1, 5, 9, 13, 17, 0, 20, 3, 2, 4, 21, 23, 3, 6, 7, 23, 6, 5, 4, 7, 8, 4, 6, 8, 17, 6, 7, 9, 8, 11, 10, 20, 13, 12, 23, 14, 13, 16, 17, 16, 15, 14, 17, 18, 14, 16, 18, 7, 17, 16, 19, 18, 1, 11, 21, 22, 3, 20, 22, 23, 13, 20, 21, 23, 21, 22, 3, 13, 4, 14]

# Manual connections
        self.m_edge_origins = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 5, 9, 10, 10, 11, 11, 12, 13, 9, 13, 14, 14, 15, 15, 16, 17, 13, 17, 18, 17, 18, 19, 19, 20]
        self.m_edge_ends = [1, 5, 17, 0, 2, 1, 3, 2, 4, 3, 0, 6, 5, 7, 6, 8, 7, 5, 9, 10, 9, 11, 10, 12, 11, 9, 13, 14, 13, 15, 14, 16, 15, 13, 17, 18, 17, 0, 19, 18, 20, 19]

    def __call__(self, sample):
        # Index finger
     #   graph_x_ = torch.tensor(np.vstack((sample['data_index'], sample['data_middle'], sample['data_thumb'])), dtype=torch.float).transpose(0, 1)
        graph_x_ = torch.tensor(np.vstack((sample['WRIST'], sample['THUMB_CMC'], sample['THUMB_MCP'], sample['THUMB_IP'], sample['THUMB_TIP'], sample['INDEX_FINGER_MCP'], sample['INDEX_FINGER_PIP'], sample['INDEX_FINGER_DIP'], sample['INDEX_FINGER_TIP'], sample['MIDDLE_FINGER_MCP'],  sample['MIDDLE_FINGER_PIP'], sample['MIDDLE_FINGER_DIP'], sample['MIDDLE_FINGER_TIP'], sample['RING_FINGER_MCP'], sample['RING_FINGER_PIP'], sample['RING_FINGER_DIP'], sample['RING_FINGER_TIP'],  sample['PINKY_MCP'],  sample['PINKY_PIP'],  sample['PINKY_DIP'],  sample['PINKY_TIP'])), dtype=torch.float).transpose(0, 1)
        
        graph_edge_index_ = torch.tensor([self.m_edge_origins, self.m_edge_ends], dtype=torch.long)
        #graph_pos_ = torch.tensor(np.vstack((self.m_taxels_x, self.m_taxels_y, self.m_taxels_z)), dtype=torch.float).transpose(0, 1)
        graph_y_ = torch.tensor([sample['grasp_type']], dtype=torch.long)

        #data_ = Data(x=graph_x_, edge_index=graph_edge_index_, pos=graph_pos_, y=graph_y_)
        data_ = Data(x=graph_x_, edge_index=graph_edge_index_, y=graph_y_)
        return data_

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
 
