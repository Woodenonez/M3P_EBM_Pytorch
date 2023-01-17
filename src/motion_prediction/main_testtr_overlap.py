import os, sys
from pathlib import Path
from copy import deepcopy as copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from net_module.net import UNetPlain, UNetPos
from data_handle import data_handler as dh
from data_handle import dataset as ds

import pre_load
from util import utils_test

# import warnings
# warnings.filterwarnings("error")

print("Program: testing\n")
if torch.cuda.is_available():
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
else:
    print(f'CUDA not working! Pytorch: {torch.__version__}.')
    sys.exit(0)
torch.cuda.empty_cache()

### XXX Back to energy map
def back2energy(x):
    x_original = copy(x)
    x[x_original>=1] *= -1
    x[x_original<1] = -torch.log(x[x_original<1])
    return x


### Config
root_dir = Path(__file__).parents[1]
# config_file = 'sdd_1t12_train.yml'
# ref_image_name = 'label.png'
# config_file = 'gcd_1t20_train.yml'
# ref_image_name = None
# config_file = 'sidv2c_1t10_train.yml'
# ref_image_name = None
config_file = 'ald_1t20_train.yml'
ref_image_name = 'label.png'
param = pre_load.load_param(root_dir, config_file, verbose=False)

composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = UNetPlain

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

    print(f'\r{idx}   ', end='')

    [ax.cla() for ax in axes.ravel()]
    
    img, label, traj, index, e_grid, ref = pre_load.main_test(dataset, net, idx=idx)
    try:
        prob_map = net.convert_grid2prob(e_grid.clone(), threshold=0.9, temperature=1)
    except:
        prob_map = e_grid.clone()
        prob_map[prob_map<0.011] = 0
        # e_grid = back2energy(e_grid)
    # pred_traj = utils_test.get_traj_from_pmap(prob_map)

    if ref is None:
        ref = img[-1,:,:]
    ### Normalize
    # prob_map = prob_map/torch.amax(prob_map, dim=(2,3))[:,:,None,None]

    ax1, ax2, ax3 = axes

    ax1.imshow(ref, cmap='gray')
    ax1.plot(traj[-1,0], traj[-1,1], 'ko', label='current')
    ax1.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    ax1.plot(label[:,0], label[:,1], 'r-', linewidth=4, label="ground truth")
    # ax1.plot(pred_traj[:,0], pred_traj[:,1], 'gx', label="pred")
    ax1.legend(prop={'size': 14}, loc='best')
    ax1.set_aspect('equal', 'box')

    ax2.imshow(torch.sum(e_grid[0,:], dim=0), cmap='gray')
    ax2.plot(label[:,0], label[:,1], 'r.')

    ax3.imshow(torch.clamp(torch.sum(prob_map[0,:], dim=0), max=1) + ref/2, cmap='gray')
    ax3.plot(traj[-1,0], traj[-1,1], 'wo', label='current')
    ax3.plot(traj[:-1,0], traj[:-1,1], 'w.') # past

    ax1.set_title('Real world')
    ax2.set_title('Energy grid')
    ax3.set_title('Probability map')
    # boundary_coords, obstacle_list, _ = return_Map(index)
    # graph = Graph(boundary_coords, obstacle_list)
    # graph.plot_map(ax1, clean=1)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    
    plt.draw()
    if pause_time==0:
        plt.pause(0.01)
        while not plt.waitforbuttonpress():  # XXX press a button to continue
            pass
    else:
        plt.pause(pause_time)

plt.show()
