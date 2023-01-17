import os, sys
from pathlib import Path

import torch
import torchvision

from net_module import loss_functions as loss_func
from net_module.net import UNetPlain, UNetPos, E3Net
from _data_handle_mmp import data_handler as dh

import pre_load

print("Program: training...\n")

# DATASET = 'GCD'
# MODE = 'TRAIN'  # 'TRAIN' or 'TEST'
# RUNON = 'LOCAL' # 'LOCAL' or 'REMOTE'
# PRED_RANGE = (1,20) # E.g. (1,10) means 1 to 10
# ref_image_name = None

# DATASET = 'SIDv2x'
# MODE = 'TRAIN'  # 'TRAIN' or 'TEST'
# RUNON = 'LOCAL' # 'LOCAL' or 'REMOTE'
# PRED_RANGE = (1,10) # E.g. (1,10) means 1 to 10
# ref_image_name = None

# DATASET = 'SDD'
# MODE = 'TRAIN'  # 'TRAIN' or 'TEST'
# RUNON = 'LOCAL' # 'LOCAL' or 'REMOTE'
# PRED_RANGE = (1,12) # E.g. (1,10) means 1 to 10
# ref_image_name = 'label.png'

DATASET = 'ALD'
MODE = 'TRAIN'  # 'TRAIN' or 'TEST'
RUNON = 'LOCAL' # 'LOCAL' or 'REMOTE'
PRED_RANGE = (1,20) # E.g. (1,10) means 1 to 10
ref_image_name = 'label.png'

if RUNON == 'LOCAL':
    BATCH_SIZE = 2
    num_workers = 0
else:
    BATCH_SIZE = None
    num_workers = 4

print(f'Run in the mode: {RUNON}!\n')

### Config
root_dir = Path(__file__).parents[1]

# loss = {'loss': torch.nn.BCEWithLogitsLoss(), 'metric': loss_func.loss_mae}
# loss = {'loss': loss_func.loss_nll, 'metric': loss_func.loss_mae}
loss = {'loss': loss_func.loss_enll, 'metric': loss_func.loss_mae}

Net = UNetPos
# Net = E3Net

config_file = pre_load.load_config_fname(DATASET, PRED_RANGE, MODE)
composed = torchvision.transforms.Compose([dh.ToTensor()])

### Training
pre_load.main_train(root_dir, config_file, Net=Net, transform=composed, loss=loss, num_workers=num_workers, 
                    batch_size=BATCH_SIZE, T_range=PRED_RANGE, ref_image_name=ref_image_name, runon=RUNON)

