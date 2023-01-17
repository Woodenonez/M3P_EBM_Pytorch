#%%
import sys
from typing import List

import torch
import numpy as np

from net_module import net

import matplotlib.pyplot as plt

#%%## 

#%%## Try E3Net
sample = torch.randn((1,3,800,800))
encoder_channels = [16,32,64,128,256]
decoder_channels = [256,128,64,32,16]
net1 = net.E3Net(in_channels=3, en_channels=encoder_channels, de_channels=decoder_channels, num_classes=1, out_layer=None)

multi_out:List[torch.Tensor] = net1(sample)

print(len(multi_out))
for out in multi_out:
    print(out.shape)