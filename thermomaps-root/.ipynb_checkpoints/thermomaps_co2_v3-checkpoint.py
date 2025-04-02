#!/usr/bin/env python
# coding: utf-8
import os
from os import system
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tm.core.prior import GlobalEquilibriumHarmonicPrior, UnitNormalPrior
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_model import DiffusionTrainer, SteeredDiffusionSampler
from tm.core.diffusion_process import VPDiffusion
from tm.architectures.unet_2d_mid_attn import Unet2D
#from tm.architectures.UNet2D_pbc import Unet2D
from tm.core.loader import Loader
import seaborn as sns


# Define the path to the directory you want to change to
#dirc_path = 'thermomaps-root'
#os.chdir(dirc_path)

import os
if not os.path.exists('plots/'):
    os.mkdir('plots')

highP= np.load('data/pressure_high.npy')
lowP = np.load('data/pressure_low.npy')

highP = highP[:, :, :, :]/(180/np.pi)
lowP = lowP[:, :, :, :]/(180/np.pi)

p_data_coordinate = np.concatenate((lowP, highP, ), axis=0)
p_data_coordinate.shape
# Transpose data to shape (10000, channel, 15, 15)
p_data_coordinate = p_data_coordinate.transpose(0, 3, 1, 2)
p_data_coordinate.shape


angle_p1 = np.concatenate(np.concatenate(lowP[:,:,:,:12]))
angle_p1 = np.concatenate(angle_p1)

angle_p12 = np.concatenate(np.concatenate(highP[:,:,:,:12]))
angle_p12 = np.concatenate(angle_p12)


# Initialize pressure values
pressure = np.array([0.0, 5.0])
pressure_list = np.ones(20000)
temperature_list = 5* np.ones(20000)

for i in range(2):
    pressure_list[10000*i:10000*(i+1)] = pressure[i]

# Add two additional channels for temperature and pressure
expanded_data = np.zeros((p_data_coordinate.shape[0],14, p_data_coordinate.shape[2], p_data_coordinate.shape[3]))

# Copy original data to the expanded array
expanded_data[:, :13, :, :] = p_data_coordinate
# print(expanded_data.shape)

# Set the 4th channel to 400 (temperature)
expanded_data[:, 12, :, :] = 5

# # Set the 5th channel to the pressure values
# expanded_data[:, 13, :, :] = p_data_coordinate[:, 12, :, :]

data_sh = int(expanded_data.shape[0]/2)

expanded_data[:data_sh,13,:,:] = pressure[0]
expanded_data[data_sh:,13,:,:] = pressure[1]

# Create trajectory dictionary
trajectoryP = {'coordinate': [], 'state_variables': [],'state_variables_P': []}
trajectoryP['coordinate'] = expanded_data[:, :14, :, :]
trajectoryP['state_variables_P'] = pressure_list
trajectoryP['state_variables'] = temperature_list


train_loader = Loader(data=trajectoryP['coordinate'], pressures =trajectoryP['state_variables_P'][:,None] , temperatures=trajectoryP['state_variables'][:,None],control_dims=(12,14))#, **TMLoader_kwargs)
prior = GlobalEquilibriumHarmonicPrior(shape=train_loader.data.shape, channels_info={"coordinate": [0,1,2,3,4,5,6,7,8,9,10,11], "fluctuation": [12], "mean": [13]})
model = Unet2D(dim=16, dim_mults=(1, 2, 4), resnet_block_groups=8, channels=14)


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

trainer.train(250, loss_type="smooth_l1", batch_size=128)



# *********************************************************************************
#                               Sampling
# *********************************************************************************
sampler = SteeredDiffusionSampler(diffusion,
                                  backbone,
                                  train_loader,
                                  prior,
                                  pred_type='x0', # must be the same as in DiffusionTrainer
                                  )


num_samp = 500

high_samples = sampler.sample_loop(num_samples=num_samp, batch_size=32, pressure=4 , temperature=1)
low_samples = sampler.sample_loop(num_samples=num_samp, batch_size=32, pressure=0 , temperature=1)


def convert_data_visual(data):
    if not isinstance(data, np.ndarray):
        data = data.numpy()
    samples = np.concatenate(np.concatenate(data[:,:12,:,:])*180/np.pi)
    samples = np.concatenate(samples)
    
    return samples    




high_samples = high_samples.numpy()
low_samples = low_samples.numpy()

x_p1 = np.concatenate(np.concatenate(low_samples[:,:12,:,:])*180/np.pi)
x_p1 = np.concatenate(x_p1)
x_p12 = np.concatenate(np.concatenate(high_samples[:,:12,:,:])*180/np.pi)
x_p12 = np.concatenate(x_p12)

plt.figure(figsize=(5,4.5))
sns.kdeplot(x_p1,label='Low P', fill=True)#, color="blue")
sns.kdeplot(x_p12,label='High P', fill=True)#, color="blue")
plt.xlim([-50,150])
plt.legend()
plt.xlabel('angle (not scaled)')
plt.savefig('plots/distribution_t5.pdf')


# Temperature depedent 
temp=[1,3,5,7,10]
high_samp=[]
low_samp=[]

for i in temp: 
    locals()['high_samples_t_{0}'.format(i)] = sampler.sample_loop(num_samples=num_samp, batch_size=32, pressure=5 , temperature=i)
    locals()['low_samples_t_{0}'.format(i)] = sampler.sample_loop(num_samples=num_samp, batch_size=32, pressure=0 , temperature=i)

    high_samp.append(locals()['high_samples_t_{0}'.format(i)])
    low_samp.append(locals()['low_samples_t_{0}'.format(i)])
    
high_samp = np.array(high_samp)
low_samp = np.array(low_samp)
    
# high_samp = np.array([high_samples_t_1,high_samples_t_3,high_samples_t_5,high_samples_t_7,high_samples_t_10])
# low_samp = np.array([low_samples_t_1,low_samples_t_3,low_samples_t_5,low_samples_t_7,low_samples_t_10])

np.save('data/high_samp.npy',high_samp)
np.save('data/low_samp.npy',low_samp)

# Pressure dependent 
presL=[0,10,20]
presH=[30,40,50]

highP_samp = []
lowP_samp = []

for i ,p1 in enumerate(presL): 
    p12  =presH[i]
    
    locals()['high_samples_p_{0}'.format(i)] = sampler.sample_loop(num_samples=num_samp, batch_size=32, pressure=p12/10 , temperature=1)
    locals()['low_samples_p_{0}'.format(i)] = sampler.sample_loop(num_samples=num_samp, batch_size=32, pressure=p1/10 , temperature=1)
    highP_samp.append(locals()['high_samples_p_{0}'.format(i)])
    lowP_samp.append(locals()['low_samples_p_{0}'.format(i)])


highP= np.load('data/pressure_high.npy')
lowP = np.load('data/pressure_low.npy')

highP_denisty = highP[400::20, :, :, :12]/(180/np.pi)
highP_denisty = highP_denisty.transpose(0,3,1,2)
lowP_denisty = lowP[400::20, :, :, :12]/(180/np.pi)
lowP_denisty = lowP_denisty.transpose(0,3,1,2)



# highP_samp = np.array([high_samples_p_5,high_samples_p_10,high_samples_p_15,high_samples_p_20,])
# lowP_samp = np.array([low_samples_p_5,low_samples_p_10,low_samples_p_15,low_samples_p_20])

highP_samp = np.array(highP_samp)
lowP_samp = np.array(lowP_samp)


np.save('data/highP_samp.npy',highP_samp)
np.save('data/lowP_samp.npy',lowP_samp)


# Pressure dependent 
plt.figure(figsize=(10,4.5))
plt.subplot(1,2,1)
sns.kdeplot(convert_data_visual(highP_denisty),label='ref', fill=False, color="blue")
for i,sample in enumerate(highP_samp):
    sns.kdeplot(convert_data_visual(sample),label='P=%i'%(presH[i]/10), fill=True)#, color="blue")
plt.legend()
plt.xlabel('angle (not scaled)')

plt.subplot(1,2,2)
sns.kdeplot(convert_data_visual(lowP_denisty),label='ref', fill=False, color="red")
for i,sample in enumerate(lowP_samp):
    sns.kdeplot(convert_data_visual(sample),label='P=%i'%(presL[i]/10), fill=True)#, color="blue")
plt.legend()
plt.xlabel('angle (not scaled)')
plt.savefig('plots/pressure_dependent.pdf')

# Temperature dependent 
plt.figure(figsize=(10,4.5))
plt.subplot(1,2,1)
sns.kdeplot(convert_data_visual(highP_denisty),label='ref', fill=False, color="blue")
for i,sample in enumerate(high_samp):
    sns.kdeplot(convert_data_visual(sample),label=temp[i], fill=True)#, color="blue")
plt.legend()
plt.xlabel('angle (not scaled)')

plt.subplot(1,2,2)
sns.kdeplot(convert_data_visual(lowP_denisty),label='ref', fill=False, color="red")
for i,sample in enumerate(low_samp):
    sns.kdeplot(convert_data_visual(sample),label=temp[i], fill=True)#, color="blue")
plt.legend()
plt.xlabel('angle (not scaled)')
plt.savefig('plots/temperature_dependent.pdf')



high_samples = high_samples_t_5
low_samples = low_samples_t_5

plt.figure(figsize=(20,10))
for j in range(12):
    # for i in range(1):
    plt.subplot(4,6,j+1)
    plt.imshow(high_samples[8,j,:,:]*180/np.pi,vmin=0,vmax=90)
    if j==0:  plt.title('high P sample angles')
    plt.colorbar()

# plt.figure(figsize=(20,5))
for j in range(12):
    # for i in range(1):
    plt.subplot(4,6,j+13)
    plt.imshow(low_samples[8,j,:,:]*180/np.pi,vmin=0,vmax=90)
    if j==0: plt.title('low P sample angles')
    plt.colorbar()
plt.tight_layout() 
plt.savefig('plots/low_distribution.pdf')


# Temperature dependent 
plt.figure(figsize=(10,4.5))
plt.subplot(1,2,1)
sns.kdeplot(convert_data_visual(highP_denisty),label='ref', fill=False, color="blue")
sns.kdeplot(convert_data_visual(high_samples_t_5),label=temp[i], fill=True)#, color="blue")
plt.legend()
plt.xlabel('angle (not scaled)')

plt.subplot(1,2,2)
sns.kdeplot(convert_data_visual(lowP_denisty),label='ref', fill=False, color="red")
sns.kdeplot(convert_data_visual(low_samples_t_5),label=temp[i], fill=True)#, color="blue")
plt.legend()
plt.xlabel('angle (not scaled)')
plt.savefig('plots/temperature_5_dependent.pdf')






