import os
import sys
import math

import torch

from util.datatype import *

def get_weight(grid:torch.Tensor, coords:torch.Tensor, sigmas:Indexable, rho=0, normalized=True):
    '''
    Description
        :Create a stack of ground truth masks to compare with the generated (energy) grid.
    Argument
        :grid - With size (BxTxHxW), the generated energy grid, used as the template.
        :coords - With size (BxTxDo), T is the pred_len, Do is the output dimension (normally 2).
        :sigmas - A tuple or list of sigma_x and sigma_y.
        :rho - The correlation parameter, currently 0.
        :normalized - Normalize the weight mask to 1 (then the weight mask is not a probability distribution anymore).
    Return
        :weight - With size (BxTxHxW), the same size as the grid.
    '''
    bs, T, H, W = grid.shape # batch_size, pred_offset, height, width
    coords = coords[:,:,:,None,None]

    sigma_x, sigma_y = sigmas[0], sigmas[1]
    x = torch.arange(0, W, device=grid.device)
    y = torch.arange(0, H, device=grid.device)
    try:
        x, y = torch.meshgrid(x, y, indexing='xy')
    except:
        y, x = torch.meshgrid(y, x) # indexing is 'ij', this is because the old torch version doesn't support indexing
    x, y = x.unsqueeze(0).repeat(bs,T,1,1), y.unsqueeze(0).repeat(bs,T,1,1)
    in_exp = -1/(2*(1-rho**2)) * ((x-coords[:,:,0])**2/(sigma_x**2) 
                                + (y-coords[:,:,1])**2/(sigma_y**2) 
                                - 2*rho*(x-coords[:,:,0])/(sigma_x)*(y-coords[:,:,1])/(sigma_y))
    weight = 1/(2*math.pi*sigma_x*sigma_y*torch.sqrt(torch.tensor(1-rho**2))) * torch.exp(in_exp)
    if normalized:
        weight = weight/(weight.amax(dim=(2,3))[:,:,None,None])
        weight[weight<0.1] = 0
    return weight

def loss_nll(data, label, sigmas:Indexable=[10,10], l2_factor:float=0.00, inputs:DebugTemp=None):
    #TODO Double check the index is (i,j) or (x,y)
    r'''
    Argument
        :data  - (BxTxHxW), the energy grid
        :label - (BxTxDo), T:pred_len, Do: output dimension [label should be the index (i,j) meaning which grid cell to choose]
    '''

    weight = get_weight(data, label, sigmas=sigmas) # Gaussian fashion [BxTxHxW]

    # XXX
    # print(f'Input size: {inputs.shape}; Energy grid size: {data.shape}; Weight grid size: {weight.shape}')
    # _, axes = plt.subplots(4,4)
    # axes[0,0].imshow(inputs[0,0].detach().cpu())
    # axes[0,1].imshow(inputs[0,1].detach().cpu())
    # axes[0,2].imshow(inputs[0,2].detach().cpu())
    # axes[0,3].imshow(inputs[0,3].detach().cpu())
    # axes[1,0].imshow(inputs[0,4].detach().cpu())
    # axes[1,1].imshow(inputs[0,5].detach().cpu())
    # axes[1,2].imshow(inputs[0,6].detach().cpu())
    # axes[1,3].imshow(inputs[0,7].detach().cpu())
    # axes[2,0].imshow(inputs[0,8].detach().cpu())
    # axes[2,1].imshow(inputs[0,9].detach().cpu())

    # axes[2,2].imshow(data[0,0].detach().cpu())
    # axes[2,3].imshow(weight[0,0].detach().cpu())
    # axes[2,3].plot(label.cpu()[0,0,0],label.cpu()[0,0,1],'rx')
    
    # plt.show()
    # XXX

    numerator_in_log   = torch.logsumexp(-data+torch.log(weight), dim=(2,3))
    denominator_in_log = torch.logsumexp(-data, dim=(2,3))

    l2 = torch.sum(torch.pow(data,2),dim=(2,3)) / (data.shape[2]*data.shape[3])
    nll = - numerator_in_log + denominator_in_log + l2_factor*l2
    if len(label.shape) == 3:
        nll = torch.sum(nll, dim=1)
    return torch.mean(nll)

def loss_enll(data, label, sigmas:Indexable=[10,10], l2_factor:float=0.00):
    r'''
    data is the energy grid, label should be the index (i,j) meaning which grid cell to choose
    :data  - BxCxHxW
    :label - BxTxDo,   T:pred_len, Do: output dimension
    '''

    weight = get_weight(data, label, sigmas=sigmas) # Gaussian fashion [BxTxHxW]

    numerator_in_log   = torch.log(torch.sum(data*weight, dim=(2,3)))
    denominator_in_log = torch.log(torch.sum(data, dim=(2,3)))

    l2 = torch.sum(torch.pow(data,2),dim=(2,3)) #/ (data.shape[2]*data.shape[3])
    nll = - numerator_in_log + denominator_in_log + l2_factor*l2
    if len(label.shape) == 3:
        nll = torch.sum(nll, dim=1)
    return torch.mean(nll)


def loss_mse(data, labels): # for batch
    # data, labels - BxMxC
    squared_diff = torch.square(data-labels)
    squared_sum  = torch.sum(squared_diff, dim=2) # BxM
    loss = squared_sum/data.shape[0] # BxM
    return loss

def loss_msle(data, labels): # for batch
    # data, labels - BxMxC
    squared_diff = torch.square(torch.log(data)-torch.log(labels))
    squared_sum  = torch.sum(squared_diff, dim=2) # BxM
    loss = squared_sum/data.shape[0] # BxM
    return loss

def loss_mae(data, labels): # for batch
    # data, labels - BxMxC
    abs_diff = torch.abs(data-labels)
    abs_sum  = torch.sum(abs_diff, dim=2) # BxM
    loss = abs_sum/data.shape[0] # BxM
    return loss
