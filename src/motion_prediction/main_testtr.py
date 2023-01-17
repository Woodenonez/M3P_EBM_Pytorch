import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from net_module.net import UNet
from _data_handle_mmp import data_handler as dh
from _data_handle_mmp import dataset as ds
from net_module import loss_functions as loss_func

import pre_load
from util import utils_test

print("Program: testing...\n")
pre_load.check_device()

### Config
config_file = 'sid_1t10_test.yml'
ref_image_name = None

root_dir = pathlib.Path(__file__).resolve().parents[1]
param = pre_load.load_param(root_dir, config_file, verbose=False)

composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = UNet

### Prepare
dataset, _, net = pre_load.main_test_pre(root_dir, config_file, composed, Net, ref_image_name=ref_image_name)

### Visualization option
idx_start = 0 # 250
idx_end = len(dataset)
pause_time = 0.1

### Visualize
fig, axes = plt.subplots(2,3)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    print(idx)

    [ax.cla() for ax in axes.ravel()]
    
    img, label, traj, index, e_grid, ref = pre_load.main_test(dataset, net, idx=idx)
    prob_map = loss_func.convert_grid2prob(e_grid.clone(), threshold=0.1, temperature=0.5)
    pred_traj = utils_test.get_traj_from_pmap(prob_map)
    if ref is None:
        ref = img[-1,:,:]

    goal_samples = utils_func.ynet_TTST(prob_map[:,-1,:,:].unsqueeze(1), 20)

    for ax in [axes[0,0], axes[1,0]]:
        ax.imshow(ref, cmap='gray')
        ax.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
        ax.plot(label[:,0], label[:,1], 'bo', label="ground truth")
        ax.plot(pred_traj[:,0], pred_traj[:,1], 'gx', label="pred")
        ax.legend(prop={'size': 14}, loc='best')
        ax.set_aspect('equal', 'box')

    for ax in axes.ravel():
        ax.plot(traj[-1,0], traj[-1,1], 'ko', label='current')

    axes[0,1].imshow(e_grid[0,4,:,:], cmap='gray')
    axes[0,1].plot(label[4,0], label[4,1], 'rx')
    axes[0,2].imshow(e_grid[0,8,:,:], cmap='gray')
    axes[0,2].plot(label[8,0], label[8,1], 'rx')

    axes[1,1].imshow(prob_map[0,4,:,:], cmap='gray')
    axes[1,1].plot(label[4,0], label[4,1], 'rx')
    axes[1,2].imshow(prob_map[0,8,:,:], cmap='gray')
    axes[1,2].plot(label[8,0], label[8,1], 'rx')

    # axes[1,3].plot(goal_samples[:,0,0,0], goal_samples[:,0,0,1], 'x')

    axes[0,0].set_title('Real world')
    axes[1,0].set_title('Real world')
    axes[0,1].set_title('Energy grid 5')
    axes[1,1].set_title('Probability map 5')
    axes[0,2].set_title('Energy grid 9')
    axes[1,2].set_title('Probability map 9')
    # boundary_coords, obstacle_list, _ = return_Map(index)
    # graph = Graph(boundary_coords, obstacle_list)
    # graph.plot_map(ax1, clean=1)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    
    if pause_time==0:
        plt.pause(0.1)
        input()
    else:
        plt.pause(pause_time)

plt.show()
