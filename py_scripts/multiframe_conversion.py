#!/usr/bin/env python
# coding: utf-8

# ## Convert multiframe .pdb to .xtc + single-frame .pdb with topology

# Import required libraries

# In[ ]:


import MDAnalysis
import os 


# If the ensemble's information is given through a single multiframe .pdb file, it must be converted using `multiframe_pdb_to_xtc`. This function takes the .pdb file as an input, and return a .xtc trajectory file together with a single-frame .pdb file, containing the ensemble's topological information. The produced pair of files can be now introduced as an input to `define_frames` to compute the corresponding samples of local and global structure.

# The arguments of `multiframe_pdb_to_xtc` are:
# * `pdb_file`: path to the multiframe .pdb file.
# * `save_file`: path where the returned files must be saved.
# * `prot_name`: a string naming the files as prot_name.xtc, prot_name.pdb.

# In[ ]:


def multiframe_pdb_to_xtc(pdb_file, save_path, prot_name):
    
    u = MDAnalysis.core.universe.Universe(pdb_file)
    at = u.atoms
    
    os.chdir(save_path)
       
    # Write the trajectory in .xtc format
    at.write(".".join([prot_name,'xtc']), frames='all')
    # Write a frame of the trajectory in .pdb format for topology information
    at.write(".".join([prot_name,'pdb']))


# ## Executing the function

# In[ ]:


'''
ensemble_name = 'my_ensemble'
pdb_multiframe =  "/path_to_multriframe_file/my_file.pdb"
save_here = "/path_where_returned_files_are_saved"

multiframe_pdb_to_xtc(pdb_file = pdb_multiframe, save_path = save_here, prot_name = ensemble_name)
'''

