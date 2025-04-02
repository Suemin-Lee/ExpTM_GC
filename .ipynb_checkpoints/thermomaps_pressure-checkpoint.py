#!/usr/bin/env python
# coding: utf-8


#%cd thermomaps-root
# ! pip -q install .
# ! pip install -qr requirements.txt
# %cd -

import os

#os.system('pip -q install .')
#os,system('pip install -qr requirements.txt')
# ! pip -q install .
# ! pip install -qr requirements.txt
# %cd -

# Define the path to the directory you want to change to
dirc_path = 'thermomaps-root'
os.chdir(dirc_path)


from os import system

system('pip -q install .')
system('pip install -qr requirements.txt')


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ising.observables import Energy, Magnetization
from ising.samplers import SwendsenWangSampler, SingleSpinFlipSampler
from ising.base import IsingModel

from data.trajectory import EnsembleTrajectory, MultiEnsembleTrajectory
from data.dataset import MultiEnsembleDataset
from data.generic import Summary

from tm.core.prior import GlobalEquilibriumHarmonicPrior, UnitNormalPrior
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_model import DiffusionTrainer, SteeredDiffusionSampler
from tm.core.diffusion_process import VPDiffusion
from tm.architectures.unet_2d_mid_attn import Unet2D
#from tm.architectures.UNet2D_pbc import Unet2D


pressure = np.array([1.0,10000.0])
p_data_coordinate = np.load('data/concatenate_all_data.npy')

p_data_coordinate = p_data_coordinate.transpose(3, 0, 1, 2)

# Verify the new shape
print(p_data_coordinate.shape)

pressure_list = np.ones(4040)

for i in range(2):
    pressure_list[1010*(i*2):1010*((i+1)*2)] = pressure[i]

trajectoryP={'coordinate':[],'state_variables':[]}

trajectoryP['coordinate'] = p_data_coordinate
trajectoryP['state_variables'] = pressure_list


from tm.core.loader import Loader
train_loader = Loader(data=trajectoryP['coordinate'], temperatures=trajectoryP['state_variables'][:,None], control_dims=(3,4))#, **TMLoader_kwargs)


prior = GlobalEquilibriumHarmonicPrior(shape=train_loader.data.shape, channels_info={"coordinate": [0,1,2], "fluctuation": [3]})
model = Unet2D(dim=16, dim_mults=(1, 2, 4), resnet_block_groups=8, channels=4)


backbone = ConvBackbone(model=model,
                        data_shape=train_loader.data_dim,
                        target_shape=16,
                        num_dims=4,
                        lr=1e-3,
                        eval_mode="train",
                        self_condition=True)


diffusion = VPDiffusion(num_diffusion_timesteps=100)

if not os.path.exists('models'):
    os.mkdir('models')


trainer = DiffusionTrainer(diffusion,
                           backbone,
                           train_loader,
                           prior,
                           model_dir="thermomaps-root/models", # save models every epoch
                           pred_type="x0", # set to "noise" or "x0"
#                            test_loader=test_loader # optional
                           )



trainer.train(500, loss_type="smooth_l1", batch_size=16)


# *********************************************************************************
#                               Sampling
# *********************************************************************************


sampler = SteeredDiffusionSampler(diffusion,
                                  backbone,
                                  train_loader,
                                  prior,
                                  pred_type='x0', # must be the same as in DiffusionTrainer
                                  )


num_samp = 5000
samples = sampler.sample_loop(num_samples=num_samp, batch_size=16, temperature=1)

import torch

output_tensor = torch.where((samples < 0) | (samples > 5), torch.zeros_like(samples), samples)
np_samples = output_tensor.numpy()

pca_size = np_samples.shape[0]

if not os.path.exists('plots/'):
    os.mkdir('plots')

for i in range(10):
    tmp = np_samples[i+1,0,:,:]
    plt.figure()
    plt.imshow(tmp)
    plt.colorbar()
    plt.savefig('plots/sample_contact_map_%i'%i+'.pdf')



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# PCA 
print(np_samples[:,:3,:,:])
tmp = np_samples[:,:3,:,:].reshape(pca_size,-1)
contact_map =  tmp.reshape(pca_size,-1)

# Apply PCA
pca = PCA(n_components=2)  # Adjust n_components as needed
principal_components = pca.fit_transform(contact_map)

# The result
print("Principal components:\n", principal_components)
print(principal_components.shape)

plt.scatter(principal_components[:,0],principal_components[:,1],s=1)


# Generate free energy surfaces 

beta=1
FW_counts,FW_xbins,FW_ybins,images = plt.hist2d(principal_components[:,0],principal_components[:,1],bins=50)
FW_counts[FW_counts==0]=FW_counts[FW_counts!=0].min()
FW_G=-np.log(FW_counts)/beta
FW_G=FW_G-np.nanmin(FW_G)

plt.figure()
plt.contourf(FW_G.transpose(),levels=5,extent=[FW_xbins[0],FW_xbins[-1],FW_ybins[0],FW_ybins[-1]],cmap='jet')
cb = plt.colorbar()
cb.set_label('density',fontsize=14)

plt.savefig('plots/pca_results.pdf')

