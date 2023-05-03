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
        
    def calculate_data_vector(self, params_values, baryon_scenario=None):        
        likelihood   = self.model.likelihood[self.likelihood]
        input_params = self.model.parameterization.to_input(params_values)
        self.model.provider.set_current_input_params(input_params)
        for (component, index), param_dep in zip(self.model._component_order.items(), 
                                                 self.model._params_of_dependencies):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(want_derived=False,
                                         dependency_params=depend_list, cached=False, **params)
        if baryon_scenario is None:
            data_vector = likelihood.get_datavector(**input_params)
        else:
            data_vector = likelihood.compute_barion_datavector_masked_reduced_dim(baryon_scenario, **input_params)
        return np.array(data_vector)

configfile = sys.argv[1]
cocoa_model = CocoaModel(configfile, 'des_y3.des_cosmic_shear')

params_fid = {'As_1e9': 2.04, 'ns': 0.97, 'H0': 70., 'omegab': 0.0495, 'omegam': 0.3,
              'w': -1., 'w0pwa': -1.,
             'DES_A1_1': 0.5, 'DES_A1_2': 0.,
             'DES_A2_1': 0., 'DES_A2_2': 0., 'DES_BTA_1': 0.0 # Restrict to the NLA model
             }

N_SRC_BINS = 4

for i in range(N_SRC_BINS):
    params_fid['DES_DZ_S%d'%(i+1)] = 0.
    params_fid['DES_M%d'%(i+1)] = 0.

print("Number of dimensions: %d"%(len(params_fid)))    
for x in params_fid:
    print(x + " : %2.3f"%(params_fid[x]))
start_time  = time.time() 
data_vector = cocoa_model.calculate_data_vector(params_fid, None)
end_time    = time.time()
print("Time taken for data vector calculation: %2.2f s "%(end_time - start_time))

dv_file = np.array([np.arange(len(data_vector)), data_vector]).T


