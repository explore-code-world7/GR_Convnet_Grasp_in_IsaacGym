import random

import numpy as np
import torch
import torch.utils.data


# inherits from GraspDatasetBase
# infact 3D map can be learned in simulation environment,which boasts plenty of datasets!

# 
class Grasp_Rgbd_TestSet(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """
    def __init__(self, device):
        self.sample_list = None

    # # concatenate numpy array
    # def add_sample(self, rgb_data, depth_data):
    #     sample = np.concatenate((rgb_data, depth_data), axis=0)
    #     self.sample_list.append(sample)

    def update(self, data):
        del self.sample_list    # 删除变量对对象的引用，对象计数=0自动释放内存
        self.sample_list = torch.from_numpy(data).float()

    # 从numpy转为tensor is unefficient
    def __getitem__(self, idx):
        return self.sample_list[idx]
    
    def __len__(self):
        return len(self.sample_list)



