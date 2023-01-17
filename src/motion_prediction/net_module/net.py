import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from net_module.backbones import *

from util.datatype import *
import matplotlib.pyplot as plt


class UNetPlain(nn.Module):
    '''
    Description
        :A plain/original UNet implementation.
    Comment
        :The input size is (batch x channel x height x width).
    '''
    def __init__(self, in_channels, num_classes, with_batch_norm=True, bilinear=True, lite:bool=True):
        super(UNetPlain,self).__init__()
        self.unet = UNet(in_channels, num_classes, with_batch_norm, bilinear, lite=lite)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        logits = self.unet(x)
        return logits

    def to_energy_grid(self, output:torch.Tensor):
        return output

    def to_prob_map(self, output:torch.Tensor, threshold=0.9, temperature=1):
        '''
        Description
            :Convert NN output (BxTxHxW) to probablity maps. High energy means low probability.
        Argument
            :output - The energy grid.
            :threshold - Within (0,1], ignore too large energy (as infinity). If 1, accept all values.
            :temperature - The temperature from energy grid to probability map.
        Example
            - For a grid !E and threshold !a, if e_ij>e_thre, e_ij=Inf, where e_thre=(e_max-e_min)*a+e_min.
        '''
        grid = output.clone()

        energy_min = torch.amin(grid, dim=(2,3))
        energy_max = torch.amax(grid, dim=(2,3))
        energy_thre = energy_min + threshold * (energy_max - energy_min) # shape (bs*T)
        grid[grid>energy_thre[:,:,None,None]] = torch.tensor(float('inf'))

        grid = grid / torch.abs(torch.amin(grid, dim=(2,3)))[:,:,None,None] # avoid too large/small energy

        grid_exp = torch.exp(-temperature*grid)
        prob_map = grid_exp / torch.sum(grid_exp.view(grid.shape[0], grid.shape[1], -1), dim=-1, keepdim=True).unsqueeze(-1).expand_as(grid)

        if torch.any(torch.isnan(prob_map)):
            raise ValueError('Nan in probability map!')
        return prob_map


class UNetPos(nn.Module):
    '''
    Description
        :A modified UNet implementation with an output layer that only outputs positive values.
        :The output layer can be 'softplus', 'poselu'.
    '''
    def __init__(self, in_channels, num_classes, with_batch_norm=True, bilinear=True, lite:bool=True, out_layer:str='poselu'):
        super(UNetPos,self).__init__()
        if out_layer.lower() not in ['softplus', 'poselu']:
            raise ValueError(f'The output layer [{out_layer}] is not recognized.')
        self.out_layer = out_layer
        self.unet = UNet(in_channels, num_classes, with_batch_norm, bilinear, lite=lite)
        if out_layer == 'softplus':
            self.outl = torch.nn.Softplus()
        elif out_layer == 'poselu':
            self.outl = PosELU(1e-6)

    def forward(self, x):
        logits = self.unet(x)
        return self.outl(logits)

    def to_energy_grid(self, output:torch.Tensor):
        if self.out_layer == 'softplus':
            energy_grid = -torch.log(torch.exp(output)-1)
        elif self.out_layer == 'poselu':
            energy_grid = output.clone()
            energy_grid[output>=1] *= -1
            energy_grid[output<1] = -torch.log(energy_grid[output<1])
        return energy_grid

    def to_prob_map(self, output:torch.Tensor, threshold=0.99, temperature=1):
        '''
        Description
            :Convert NN output (BxTxHxW) to probablity maps. High energy means low probability.
        Argument
            :output - The processed energy grid, i.e. after the positive output layer.
            :threshold - Within (0,1], ignore too large energy (too small processed energy). If 1, accept all values.
            :temperature - The temperature from energy grid to probability map.
        Example
            - For a processed grid !E' and threshold !a, if e'_ij<e'_thre, e'_ij=0, where e'_thre=e'_max*(1-a).
        '''
        grid_pos = output.clone()

        pos_energy_max = torch.amax(grid_pos, dim=(2,3))
        pos_energy_thre = (1-threshold) * pos_energy_max # shape (bs*T)
        grid_pos[grid_pos<pos_energy_thre[:,:,None,None]] = torch.tensor(0.0)

        numerator = torch.exp(torch.tensor(temperature))*grid_pos
        denominator = torch.sum(numerator.view(grid_pos.shape[0], grid_pos.shape[1], -1), dim=-1, keepdim=True).unsqueeze(-1).expand_as(grid_pos)
        prob_map = numerator / denominator

        if torch.any(torch.isnan(prob_map)):
            raise ValueError('Nan in probability map!')
        return prob_map


class E3Net(nn.Module): # 
    '''
    Ongoing, the idea is to have an Early Exit Energy-based (E3) network.
    
    Comment
        :The input size is (batch x channel x height x width).
    '''
    def __init__(self, in_channels, num_classes, en_channels, de_channels, with_batch_norm=False, out_layer:str='softplus'):
        super(E3Net,self).__init__()
        if (out_layer is not None): 
            if (out_layer.lower() not in ['softplus', 'poselu']):
                raise ValueError(f'The output layer [{out_layer}] is not recognized.')

        self.encoder = UNetTypeEncoder(in_channels, en_channels, with_batch_norm)
        self.inc = DoubleConv(en_channels[-1], out_channels=en_channels[-1], with_batch_norm=with_batch_norm)

        up_in_chs  = [en_channels[-1]] + de_channels[:-1]
        up_out_chs = up_in_chs # for bilinear
        dec_in_chs  = [enc + dec for enc, dec in zip(en_channels[::-1], up_out_chs)] # add feature channels
        dec_out_chs = de_channels

        self.decoder = nn.ModuleList()
        for in_chs, out_chs in zip(dec_in_chs, dec_out_chs):
            self.decoder.append(UpBlock(in_chs, out_chs, bilinear=True, with_batch_norm=with_batch_norm))
        
        self.multi_out_cl = nn.ModuleList() # out conv layer
        for de_ch in de_channels:
            self.multi_out_cl.append(nn.Conv2d(de_ch, num_classes, kernel_size=1))

        if out_layer == 'softplus':
            self.outl = torch.nn.Softplus()
        elif out_layer == 'poselu':
            self.outl = PosELU(1e-6)
        else:
            self.outl = torch.nn.Identity()

    def forward(self, x):
        features:list = self.encoder(x)

        features = features[::-1]
        x = self.inc(features[0])

        multi_out = []
        for feature, dec, out in zip(features[1:], self.decoder, self.multi_out_cl):
            x = dec(x, feature)
            multi_out.append(self.outl(out(x)))
        return multi_out


### XXX
class UNetLite_PELU(nn.Module):
    # batch x channel x height x width
    def __init__(self, in_channels, num_classes=1, with_batch_norm=True, bilinear=True, axes=None):
        super(UNetLite_PELU,self).__init__()

        self.inc = DoubleConv(in_channels, 16, with_batch_norm=with_batch_norm)
        self.down1 = DownBlock(16, 32, with_batch_norm=with_batch_norm)
        self.down2 = DownBlock(32, 64, with_batch_norm=with_batch_norm)
        self.down3 = DownBlock(64, 128, with_batch_norm=with_batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(128, 256 // factor, with_batch_norm=with_batch_norm)
        self.up1 = UpBlock(256, 128 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up2 = UpBlock(128, 64 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up3 = UpBlock(64, 32 // factor, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.up4 = UpBlock(32, 16, bilinear=bilinear, with_batch_norm=with_batch_norm)
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

        self.outl = PosELU(1e-6)

        self.axes = axes

    def forward(self, x):
        # _, [ax1,ax2] = plt.subplots(1,2); ax1.imshow(self.outl(logits)[0,-1,:].detach().cpu()), ax2.imshow(x[0,-2,:].detach().cpu())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x0 = self.up1(x5, x4)
        x0 = self.up2(x0, x3)
        x0 = self.up3(x0, x2)
        x0 = self.up4(x0, x1)
        logits = self.outc(x0)
        return self.outl(logits)