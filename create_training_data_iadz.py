import sys
import numpy as np
import time 
import h5py as h5
import torch

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model
from cobaya.conventions import kinds, _timing, _params, _prior, _packages_path

def get_model(yaml_file):
    info  = yaml_load_file(yaml_file)
    updated_info = update_info(info)
    model =  Model(updated_info[_params], updated_info[kinds.likelihood],
               updated_info.get(_prior), updated_info.get(kinds.theory),
               packages_path=info.get(_packages_path), timing=updated_info.get(_timing),
               allow_renames=False, stop_at_error=info.get("stop_at_error", False))
    return model

class CocoaModel:
    def __init__(self, configfile, likelihood):
        self.model      = get_model(configfile)
        self.likelihood = likelihood
        
    def calculate_data_vector(self, params_values_list, baryon_scenario=None):        
        likelihood   = self.model.likelihood[self.likelihood]
        data_vector_list = []
        for params_values in params_values_list:
            start_time = time.time()
            input_params = self.model.parameterization.to_input(params_values)
            self.model.provider.set_current_input_params(input_params)
            for (component, index), param_dep in zip(self.model._component_order.items(), 
                                                 self.model._params_of_dependencies):
                depend_list = [input_params[p] for p in param_dep]
                params = {p: input_params[p] for p in component.input_params}
                compute_success = component.check_cache_and_compute(want_derived=False,
                                         dependency_params=depend_list, cached=True, **params)
            data_vector = likelihood.get_datavector(**input_params)
            data_vector_list.append(data_vector)
            end_time = time.time()
            print("Time taken: %2.2f s"%(end_time - start_time))
        return data_vector_list

configfile = sys.argv[1]
cocoa_model = CocoaModel(configfile, 'des_y3.des_cosmic_shear')


N_SRC_BINS = 4

## Chainge the mean to the mean value of DZ as found in the DES-Y3 papers
dz_mean = np.zeros(N_SRC_BINS)

priors = np.array([[-5., 5.],        # min-max value of DES_A1_1
                   [-5., 5.]         # min-max value of DES_A1_2
                  ])

print("priors.shape: "+str(priors.shape))
def scale_unit_random(unit_random, priors):
    """
    Function to scale a random number between 0. and 1., then multiplies by values to get things within the prior
    """
    delta_pars = priors[:,1] - priors[:,0]
    return priors[:,0] + delta_pars * unit_random

def get_params_dict_from_unit_random(unit_random, priors):
    """
    Function that takes as its input a random number and returns a dictionary that can be used by the `calculate_data_vector` function. 
    """
    pars = scale_unit_random(unit_random, priors)
    params_fid = {'As_1e9': 2.04, 'ns': 0.97, 'H0': 70., 'omegab': 0.0495, 'omegam': 0.3,
                    'w0pwa': -1., 'w': -1., 
                    'DES_A1_1': pars[0], 'DES_A1_2': pars[1],
                    'DES_A2_1': 0., 'DES_A2_2': 0., 'DES_BTA_1': 0.0 # Restrict to the NLA model
             }
    for i in range(N_SRC_BINS):
        params_fid['DES_DZ_S%d'%(i+1)] = dz_mean[i]
        params_fid['DES_M%d'%(i+1)] = 0.

    return params_fid

N_samples = 10
N_dims    = 2

unit_random_arr = np.random.uniform(size=(N_samples, N_dims))

params_list = []
dv_list     = []

for i in range(N_samples):
    params_arr  = scale_unit_random(unit_random_arr[i], priors) 
    params     = get_params_dict_from_unit_random(unit_random_arr[i], priors) 
    params_list.append(params)
    
data_vector_list = cocoa_model.calculate_data_vector(params_list)

"""
if(i%5==0):
    # TRAINING_DV_DATA_OUTPUT: The output file where to store the data vector list for training
    # TRAINING_PARAMS_DATA_OUTPUT: The output file where to store the parameter list for training 
    np.save(TRAINING_DV_DATA_OUTPUT, np.array(dv_list))
    np.save(TRAINING_PARAMS_DATA_OUTPUT, np.array(params_list))
"""
