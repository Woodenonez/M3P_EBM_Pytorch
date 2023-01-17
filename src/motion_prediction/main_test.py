import os
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from net_module.net import UNetPlain, UNetPos, UNet
from _data_handle_mmp import dataset as ds
from _data_handle_mmp import data_handler as dh

import pre_load

print("Program: testing...\n")
pre_load.check_device()

### Config
# config_file = 'sdd_1t12_train.yml'
# ref_image_name = 'label.png'
# config_file = 'gcd_1t20_train.yml'
# ref_image_name = None
# config_file = 'sidv2x_1t10_train.yml'
# ref_image_name = None
config_file = 'ald_1t20_train.yml'
ref_image_name = 'label.png'

root_dir = pathlib.Path(__file__).resolve().parents[1]
param = pre_load.load_param(root_dir, config_file, verbose=False)

composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = UNetPos

### Prepare
dataset, _, net = pre_load.main_test_pre(root_dir, config_file, composed, Net, ref_image_name=ref_image_name)

### Visualization option
idx_start = 0 # 250
idx_end = len(dataset)
pause_time = 0

### Visualize
fig, axes = plt.subplots(1,3)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    print(f'\r{idx}', end='')

    [ax.cla() for ax in axes.ravel()]
    
    img, label, traj, index, e_grid, ref = pre_load.main_test(dataset, net, idx=idx)
    prob_map = net.convert_grid2prob(e_grid, threshold=0.1, temperature=0.5)
    # prob_map = net.convert_exp2prob(e_grid)

    for ax in axes.ravel():
        ax.plot(traj[-1,0], traj[-1,1], 'ko', label='current')

    axes[0].imshow(img[-1], cmap='gray')
    axes[0].plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    axes[0].plot(label[:,0], label[:,1], 'bo', label="ground truth")
    axes[0].legend(prop={'size': 14}, loc='best')
    axes[0].set_aspect('equal', 'box')
    axes[0].set_title('Real world')

    axes[1].imshow(prob_map[0,-1,:], cmap='gray')
    axes[1].plot(label[-1,0], label[-1,1], 'rx')
    axes[0].set_title('Last energy grid')

    axes[2].imshow(e_grid[0,-1,:], cmap='gray')
    axes[2].plot(label[-1,0], label[-1,1], 'rx')
    axes[0].set_title('Last probability map')

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    
    plt.draw()
    if pause_time==0:
        plt.pause(0.01)
        while not plt.waitforbuttonpress():
            pass
    else:
        plt.pause(pause_time)

plt.show()
