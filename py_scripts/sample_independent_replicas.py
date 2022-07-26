#!/usr/bin/env python
# coding: utf-8

# # Sampling indepedent replicas

# Import required libraries

# In[ ]:


import os
import h5py
import numpy as np


# When no independent replicas of the considered ensemble are provided, one can generate them by uniformly subsampling its conformations. This allows taking uncertainty into account and remove its effect when estimating differences between a pair of ensembles. The function `sample_ind_replicas` takes the samples of local and global structure computed with `define_frames` function, and samples a given number of independent replicas. Its arguments are:
# * `prot_name`: the ensemble's name, such that structures are saved as prot_name_coordinates.hdf5 and prot_name_dihedrals.hdf5.
# * `coordinates_path`: the path where the file prot_name_coordinates.hdf5 is located.
# * `dihedrals_path`: the path where the file prot_name_dihedrals.hdf5 is located.
# * `save_to`: the file where to save the produced replicas.
# * `N_replicas`: The number of replicas to generate. If `None`, it is asked as an input after printing the number of available conformations. 
# 
# The function prints the number of conformations that compose the given ensemble, and asks the practitioner to input the number of independent replicas to be sampled, if is has not been introduced as an argument. Then, it produces and saves the corresponding .hdf5 files.

# In[ ]:


def sample_ind_replicas(prot_name, coordinates_path, dihedrals_path, save_to, N_replicas = None):
    
    os.chdir(coordinates_path)
    h5f_1 = h5py.File("_".join([prot_name,'coordinates.hdf5']),'r')
    os.chdir(dihedrals_path)
    h5f_2 = h5py.File("_".join([prot_name,'dihedrals.hdf5']),'r')
    N_conf = np.shape(h5f_1['ensemble'])[0]
    print("".join(['\n',prot_name,' ensemble contains ',str(N_conf),' conformations.\n']))
    
    if N_replicas is None:
        N_replicas = float(input("Please introduce the number of independent replicas to sample:\n"))
    
    print('Sampling and saving files...\n')
    replicas_indices = np.random.choice(np.arange(N_replicas), size = N_conf, replace = True)
    
    os.chdir(save_to)
    for i in np.arange(N_replicas):
        h5f_1_i = h5py.File("_".join([prot_name,str(int(i)),'coordinates.hdf5']),'w')
        h5f_1_i.create_dataset("ensemble", data = h5f_1["ensemble"][replicas_indices == i,:,:])
        h5f_1_i.close()
        
        h5f_2_i = h5py.File("_".join([prot_name,str(int(i)),'dihedrals.hdf5']),'w')
        h5f_2_i.create_dataset("ensemble", data = h5f_2["ensemble"][replicas_indices == i,:,:])
        h5f_2_i.close()    


# ## Executing the function

# In[ ]:


'''
ensemble_name = 'my_ensemble'
coordinates_folder = "/path_to_coordinates_file"
dihedrals_folder = "/path_to_dihedrals_file"
save_here = "/save_replicas_here"

sample_ind_replicas(prot_name = ensemble_name, coordinates_path = coordinates_folder, dihedrals_path = dihedrals_folder, save_to = save_here)    
'''

