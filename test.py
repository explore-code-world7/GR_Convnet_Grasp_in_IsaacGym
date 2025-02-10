import argparse
import logging
import time

import numpy as np
# import torch.utils.data
from torch.utils import data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results
from grtest.rgbd_dataset import Grasp_Rgbd_TestSet

from torchvision import transforms, utils
from torchvision import datasets
import torch
import matplotlib.pyplot as plt 
# %matplotlib inline

# train_data = datasets.ImageFolder("usage/depth")      # ImageFolder文件夹名为其内的图片标签
import os
from PIL import Image
import numpy as np

# build dataset

# device = torch.device("cpu")
device = torch.device("cuda:0")

dep_folder = "grtest/depth"
rgb_folder = "grtest/rgb"

dep_list = []
rgb_list = []

for file_name in os.listdir(dep_folder):
    dep_img = Image.open(os.path.join(dep_folder, file_name))
    dep_array = np.array(dep_img)           # [224,224]
    dep_list.append(dep_array)

for file_name in os.listdir(rgb_folder):
    rgb_img = Image.open(os.path.join(rgb_folder, file_name))
    if rgb_img.mode == "RGBA":
        rgb_img = rgb_img.convert("RGB")
    rgb_array = np.array(rgb_img)           # [224,224,3]
    rgb_list.append(rgb_array)

print(len(dep_list), len(rgb_list))

test_array = [np.concatenate((np.expand_dims(dep_list[idx], 2), rgb_list[idx]),2)  for idx in range(len(dep_list))]
test_array = np.concatenate([np.expand_dims(np.transpose(array,(2, 0, 1)),0) for array in test_array], 0)       # [512, 4,224, 224]
print(test_array.shape)         # [224,224,4]


TestDataset = Grasp_Rgbd_TestSet(device)
TestDataset.update(test_array)

Test_Loader = data.DataLoader(TestDataset, batch_size = 64, shuffle =True)


# load model
from inference.models.grconvnet3 import GenerativeResnet

# gcnet = GenerativeResnet().to(device)
gcnet = torch.load("logs/250131_2338_training_cornell/epoch_44_iou_0.96").to(device)

pos_list = []
# print(Test_Loader)
tmp_list = []

# 1. 虽然label的数据转移到cpu上了,但是还存储相应的梯度
gcnet.eval()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# process output
from utils.dataset_processing import evaluation, grasp
from inference.post_process import post_process_output


# store imgs
import imageio
from skimage.feature import peak_local_max
from utils.dataset_processing.grasp import Grasp


id = 0

# def draw_rect(q_img, ang_img, lengthno_grasps=1):
#     local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)    
#     grasps = []
#     for grasp_point_array in local_max:
#         grasp_point = tuple(grasp_point_array)

#         grasp_angle = ang_img[grasp_point]

#         xo = np.cos(grasp_angle)
#         yo = np.sin(grasp_angle)

#         y1 = grasp_point[0] + self.length / 2 * yo
#         x1 = grasp_point[1] - self.length / 2 * xo
#         y2 = grasp_point[0] - self.length / 2 * yo
#         x2 = grasp_point[1] + self.length / 2 * xo

#         g = Grasp(grasp_point, grasp_angle)
#         if width_img is not None:
#             g.length = width_img[grasp_point]
#             g.width = g.length / 2

#         grasps.append(g)


for batch_idx, batch_data in enumerate(Test_Loader):
    batch_data = batch_data.to(device)
    # network.cuda()
    # batch_data = batch_data.to(device)
    with torch.no_grad():
        # label = gcnet.predict(batch_data)
        # pos, _, _, _ = gcnet(batch_data)
        q_img, cos_img, sin_img, width_img = gcnet(batch_data)
    
    # print(f"q_img.shape = {q_img.shape}")
    q_img, ang_img, width_img = post_process_output(q_img, cos_img, sin_img, width_img)
    # print(f"q_img.shape = {q_img.shape}")

    if(len(q_img.shape)==2):
        q_img = np.expand_dims(q_img, 0)
        ang_img = np.expand_dims(ang_img, 0)
        width_img = np.expand_dims(width_img, 0)

    # print(q_img.shape)
    # print(q_img.size)
    # print(len(q_img))
    # print(f"img number = {len(q_img)}")

    # why does width boasts magnitude of 1W+?
    for img_idx in range(q_img.shape[0]):
        grasps = grasp.detect_grasps(q_img[img_idx], ang_img[img_idx], np.ones((224, 224))*6, 2)
        for gid, _grasp in enumerate(grasps):
            
            print(_grasp.center, _grasp.angle)

            # draw grap rectangle
            ax.clear()
            ax.set_xlim(1,225)
            ax.set_ylim(1,225)
            _grasp.plot(ax)
            # plt.show()

            # draw grasp point
            # plane = np.zeros((224, 224))
            # plane[_grasp.center[0],_grasp.center[1]]=1
            # plt.imshow(plane)

            fig.savefig(os.path.join("grtest", "predicted", f"image_{id}_{gid}.png"))       
            # plt.close()
        id +=1

    # pos = pos.cpu()
    # pos2 = pos.clone()
    # pos2 = pos2.cpu()
    # print(pos.device)
    # pos_list.append(label["pos"])
    # label["pos"] = label["pos"].cpu()
    # label["sin"] = label["sin"].cpu()
    # label["cos"] = label["cos"].cpu()
    # label["width"] = label["width"].cpu()
    # tmp_list.append(label["pos"])
    # print(label["pos"].device)
    
    # pos_list.extend(torch.split(label["pos"], 1, dim=0))    # 1=每个子张量的大小

    # pos_list.append(batch_data)
    # pos_list.append(pos)
    # pos_list.extend(torch.split(pos, 1, dim=0))    # 1=每个子张量的大小
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))
    # print(f"batch_idx = {batch_idx}")
    # torch.cuda.empty_cache()
    # print(pos_list[-1].device)
    # del batch_data
    # print(batch_data.device)
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))
    # del label
    # gc.collect()
    # del batch_data
    # pos_list.append(batch_data)


# print(len(pos_list))
# print(pos_list[0])