import torch
import numpy as np
from typing import Any, Callable, Dict, List
import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class UnitNormalPrior:
    def __init__(self, shape, channels_info):
        self.channels_info = channels_info  # Dictionary to define channel types
        self.num_fluct_ch = len(self.channels_info['fluctuation'])
        self.num_mean_ch = len(self.channels_info['mean'])
        self.num_coord_ch = len(self.channels_info['coordinate'])
        """Initialize the Unit Normal Prior with the shape of the samples."""
        self.shape = list(shape)[1:]
        logger.debug(f"Initialized a Prior with shape {self.shape}.")
        logger.debug(f"The first dimension of the supplied {shape=} must be the batch size.")

    def sample(self, batch_size, *args, **kwargs):
        """Sample from a unit normal distribution."""
        shape = [batch_size] + self.shape
        # print(f"Sampling from a UnitNormalPrior with shape {shape}")
        return torch.normal(mean=0, std=1, size=shape)

class GlobalEquilibriumHarmonicPrior(UnitNormalPrior):
    def __init__(self, shape, channels_info):
        """Initialize GEHP with shape and channels information."""
        super().__init__(shape, channels_info)
        # self.channels_info = channels_info  # Dictionary to define channel types
        # print(len(self.channels_info['fluctuation']))
        # print(len(self.channels_info['coordinate']))

    def sample(self, batch_size, pressures, temperatures, *args, **kwargs):
        """Sample from a distribution where variance is defined by temperatures."""
        logger.debug(f"{temperatures=}")
        temperatures = torch.Tensor(np.array(temperatures))
        pressures = torch.Tensor(np.array(pressures))
        full_shape = [batch_size] + self.shape
        coord_shape = [batch_size] + [self.num_coord_ch] + self.shape[1:]
        fluct_shape = [batch_size] + [self.num_fluct_ch] + self.shape[1:]
        mean_shape = [batch_size] + [self.num_mean_ch] + self.shape[1:]
        # print(f"{full_shape=}")
        # print(f"{coord_shape=}")
        # print(f"{fluct_shape=}")
        # print(f"{mean_shape=}")
        # print('')
        # print(f"{temperatures.shape=}")
        # print(f"{pressures.shape=}")
        samples = torch.empty(full_shape)
        # print('samples shape',samples.shape)

        #assert (temperatures.shape[1] == self.num_coord_ch and 
        #        (temperatures.shape[0] == 1 or temperatures.shape[0] == batch_size)), \
        #f"{temperatures.shape=}. Expected (1,{self.num_coord_ch}) or ({batch_size}, {self.num_coord_ch})"

        temps_for_each_channel_bool = temperatures.shape[1] == self.num_coord_ch 
        single_temp_provided_bool = temperatures.shape[0] == 1
        temps_for_each_sample_in_batch_bool = temperatures.shape[0] == batch_size


        pres_for_each_channel_bool = pressures.shape[1] == self.num_coord_ch 
        single_pres_provided_bool = pressures.shape[0] == 1
        pres_for_each_sample_in_batch_bool = pressures.shape[0] == batch_size

        if not temps_for_each_channel_bool and temps_for_each_sample_in_batch_bool:
            temperatures = temperatures.unsqueeze(-1).unsqueeze(-1).expand(*coord_shape)
            coord_variances = temperatures # expand along batch and coordinate dims
        else:
            coord_variances = temperatures.unsqueeze(-1).unsqueeze(-1).expand(*coord_shape) # expand along batch and coordinate dims
        fluct_variances = torch.full((fluct_shape), 1)
        # print('coord_variances =',coord_variances.shape)
        # print('fluct_variances = ',fluct_variances.shape)
        variances = torch.cat((coord_variances, fluct_variances, fluct_variances), dim=1)
    

        if not pres_for_each_channel_bool and pres_for_each_sample_in_batch_bool:
            pressures = pressures.unsqueeze(-1).unsqueeze(-1).expand(*coord_shape)
            coord_means = pressures
        else:
            coord_means = pressures.unsqueeze(-1).unsqueeze(-1).expand(*coord_shape)
        
        fluct_means = torch.zeros(mean_shape)
        means = torch.cat((coord_means, fluct_means, fluct_means), dim=1)
    
        for sample_idx, (ch_variances, ch_means) in enumerate(zip(variances, means)):
            # print('ch_variances=',ch_variances.shape)
            # print('ch_means=',ch_variances.shape)
            # print(samples.shape)
            logger.debug(f"{ch_variances.shape}")
            samples[sample_idx] = torch.normal(mean=ch_means, std=torch.sqrt(ch_variances))
        
        logger.debug(f"{samples.shape}")
        return samples
    
    
    

