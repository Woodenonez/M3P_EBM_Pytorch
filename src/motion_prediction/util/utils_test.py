import numpy as np
import torch
import matplotlib.patches as patches


def plot_on_ref(axes, ref, traj, label, e_grid, prob_map, samlpes):
    ax1, ax2, ax3 = axes
    ax1.imshow(ref, cmap='gray')
    ax1.plot(traj[-1,0], traj[-1,1], 'mo', label='current')
    ax1.plot(traj[:-1,0], traj[:-1,1], 'm.') # past
    ax1.plot(label[:,0], label[:,1], 'ro', label="ground truth")
    ax1.legend()
    ax1.legend(prop={'size': 14}, loc='upper right')
    ax1.set_aspect('equal', 'box')

    ax2.imshow(e_grid, cmap='gray')
    ax2.plot(traj[-1,0], traj[-1,1], 'ko', label='current')
    ax2.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    ax2.plot(label[:,0], label[:,1], 'rx', label="ground truth")
    ax2.plot(samlpes[0,-1,:,0], samlpes[0,-1,:,1], 'g.')

    ax3.imshow(prob_map, cmap='hot')
    ax3.plot(traj[-1,0], traj[-1,1], 'ko', label='current')
    ax3.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    ax3.plot(label[:,0], label[:,1], 'rx', label="ground truth")

    ax1.set_title('Real world')
    ax2.set_title('Energy grid')
    ax3.set_title('Probability map')

def plot_Gaussian_ellipses(ax, mu_list, std_list, conf_list, factor=1, expand=0, alpha:float=None, label:str=None):
    for mu, std, conf in zip(mu_list, std_list, conf_list):
        patch = patches.Ellipse(mu, std[0]*factor+expand, std[1]*factor+expand, fc='y', ec='purple', alpha=alpha, label=label)
        ax.add_patch(patch)
        ax.text(mu[0], mu[1], conf)

def get_traj_from_pmap(prob_map) -> np.ndarray:
    traj = []
    prob_map = prob_map[0,:]
    if prob_map.device == 'cuda':
        prob_map = prob_map.cpu()
    for i in range(prob_map.shape[0]):
        index = torch.where(prob_map[i,:]==torch.max(prob_map[i,:]))
        try:
            index = [x.item() for x in index[::-1]]
        except:
             index = traj[-1]
        traj.append(index)
    return np.array(traj)

