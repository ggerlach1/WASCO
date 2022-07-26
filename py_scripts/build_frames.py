#!/usr/bin/env python
# coding: utf-8

# # Define ensemble's universal coordinates

# Import required libraries

# In[ ]:


import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import partial
import mdtraj as md
import h5py
import itertools
import pandas as pd
import warnings #Optional
warnings.filterwarnings("ignore") #Optional


# The function `get_coordinates` compute universal local and global coordinates for a single conformation, extracted from a .pdb or (.xtc, .top) file. Then, this function is parallelized across the entire conformational ensemble. Its arguments are
# 
# * `conf_name`: For a conformation extracted from a .pdb file, the name (string) of the .pdb file. For a conformation extracted form a (.xtc, .top) file, an integer indicating the conformation number in [0, number of conformations].
# * `pdb`: If the conformation is extracted from a .pdb file, the path to the folder containing the file. Otherwise, to be set to `None`.
# * `traj`: If the conformation is extracted from a (.xtc, .top) file, the [mdtraj](https://www.mdtraj.org) file given as ` traj = md.load_xtc(xtc_file, top = top_file)`, where `xtc_file` and `top_file` are the trajectory's .xtc and .top file, respectively (see [md.traj documentation](https://www.mdtraj.org/1.9.8.dev0/examples/introduction.html)).  Otherwise, to be set to `None`.
# 
# Normally, the user can load `get_coordinates` and skip to the next function `define_frames`, which integrates `get_coordinates` and its arguments, and should be used if coordinates for an entire ensemble need to be defined.

# In[ ]:


def get_coordinates(conf_name, pdb = None, traj = None):
    
    import numpy as np
    from Bio import PDB
    import os
    import pandas as pd
    import itertools
    
    aa_list = list(["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU","GLY","HIS", "ILE", "LEU", "LYS", "MET", "PHE","PRO", "SER", "THR", "TRP", "TYR", "VAL"])
    
    parser = PDB.PDBParser()

    def get_structure(conf_name, conf_path): 
      
        os.chdir(conf_path)
        struct = parser.get_structure('prot',conf_name)
   
        coor_x=list()
        coor_y=list()
        coor_z=list()
        model_list=list()
        chain_list=list()
        residue_list=list()
        atom_list=list()
        position_list=list()
        phi_list=list()
        psi_list=list()
        model_list_dihedrals=list()
        chain_list_dihedrals=list()
        residue_list_dihedrals=list()
        position_list_dihedrals=list()
        
        for model in struct:
            for chain in model:
                
                poly = PDB.Polypeptide.Polypeptide(chain) 
                phi_psi = poly.get_phi_psi_list()
                
                for res_index, residue in enumerate(poly) :
                    
                    phi, psi = phi_psi[res_index]
                    phi_list.append(phi)
                    psi_list.append(psi)
                    model_list_dihedrals.append(1+model.id)
                    chain_list_dihedrals.append(chain.id)
                    residue_list_dihedrals.append(residue.get_resname())
                    position_list_dihedrals.append(residue.get_full_id()[3][1])
                
                for residue in chain:
                    for atom in residue:
                        x,y,z = atom.get_coord()
                        coor_x.append(x)
                        coor_y.append(y)
                        coor_z.append(z)
                        model_list.append(1+model.id)
                        chain_list.append(chain.id)
                        residue_list.append(residue.get_resname())
                        atom_list.append(atom.id)
                        position_list.append(residue.get_full_id()[3][1])
                                       
        data = {'Model': model_list,
                'Chain': chain_list,
                'Residue': residue_list,
                'Atom': atom_list,
                'Position': position_list,
                'coor_x': coor_x,
                'coor_y': coor_y,
                'coor_z': coor_z
                }
                       
        df = pd.DataFrame (data, columns = ['Model','Chain','Residue','Atom','Position','coor_x','coor_y','coor_z'],index=None)
        df = df[df.Model == df.Model[0]] # Keep just one model
        df = df[df.Chain == df.Chain[0]] # Keep just one chain
                
        data_dihedrals = {'Model': model_list_dihedrals,
                    'Chain': chain_list_dihedrals,
                     'Residue': residue_list_dihedrals,
                     'Position': position_list_dihedrals,
                     'Phi': phi_list,
                     'Psi': psi_list
                      }
                    
        df_dihedrals = pd.DataFrame (data_dihedrals, columns = ['Model','Chain','Residue','Position','Phi','Psi'],index=None)
        df_dihedrals = df_dihedrals[df_dihedrals.Model == 1] # Keep just one model
                    
        return df, df_dihedrals[['Residue','Position','Phi','Psi']]
            
    if traj is None and pdb is not None:
   
        df, dihedrals = get_structure(conf_name, conf_path = pdb)
        L = len(np.unique(df.Position))
        
    elif pdb is None and traj is not None:
        
        traj = traj[conf_name]
        top_table = traj.top.to_dataframe()[0]
        df = pd.concat([top_table, pd.DataFrame(traj.xyz[0], columns = np.array(['x','y','z']))], axis = 1)
        df = df[['segmentID','chainID','resName','name','resSeq','x','y','z']]
        df.columns = ['Model','Chain','Residue','Atom','Position','coor_x','coor_y','coor_z']
        L = len(np.unique(df.Position))
        
        # Get dihedrals
        phi_psi = np.zeros([len(np.unique(top_table.resSeq)), 2])
        phi_psi[1:np.shape(phi_psi)[0],0] = md.compute_phi(traj)[1]
        phi_psi[0:np.shape(phi_psi)[0]-1,1] = md.compute_psi(traj)[1]
        
        res_names = np.array(top_table.resName[top_table.name=='CA'])
        res_pos = np.unique(top_table.resSeq)
        
        dihedrals = pd.DataFrame(np.concatenate([np.reshape(res_names,[np.shape(res_names)[0],1]), np.reshape(res_pos,[np.shape(res_pos)[0],1]) , phi_psi], axis = 1))
        dihedrals.columns = ['Residue','Position','Phi','Psi'] 
        dihedrals.Phi[0] = np.NaN; dihedrals.Psi[dihedrals.shape[0]-1] = np.NaN
        
    # Build reference systems
 
    basis_angles = np.array([1.917213, 1.921843, 2.493444])
    b = np.array([np.cos(basis_angles[0]), np.cos(basis_angles[1]), np.cos(basis_angles[2])]).T
    
    # 1. Definition of the reference frame on every sequence position

    CA_coor = df.loc[ (df.Atom == 'CA') , ['coor_x','coor_y','coor_z']].to_numpy() # CA coordinates 
    N_coor = df.loc[ (df.Atom == 'N')  , ['coor_x','coor_y','coor_z']].to_numpy() # N coordinates 
    C_coor = df.loc[ (df.Atom == 'C')  , ['coor_x','coor_y','coor_z']].to_numpy() # C coordinates 
    
    N_CA_coor = N_coor - CA_coor; N_CA_coor = N_CA_coor / np.linalg.norm(N_CA_coor, axis = 1)[:, None]
    C_CA_coor = C_coor - CA_coor; C_CA_coor = C_CA_coor / np.linalg.norm(C_CA_coor, axis = 1)[:, None]
    CxN_coor = np.cross(C_CA_coor, N_CA_coor); CxN_coor = CxN_coor / np.linalg.norm(CxN_coor, axis = 1)[:, None]
     
    A_list = np.concatenate([N_CA_coor,C_CA_coor,CxN_coor], axis = 1)
    A_list = np.reshape(A_list, [np.shape(A_list)[0]*3, 3])
    A_list = [A_list[i:(i+3),:] for i in 3*np.arange(np.shape(N_CA_coor)[0])]

    CB_coor = np.linalg.solve(A_list, [b for i in np.arange(len(A_list))]) # Virtual CB coordinates
 
    # Reference frames 
    
    b1_coor = CB_coor / np.linalg.norm(CB_coor, axis = 1)[:, None] # b1 = CA-CB
    CN_coor = N_CA_coor - C_CA_coor # CN
    b2_coor = np.cross(CN_coor, b1_coor); b2_coor = b2_coor / np.linalg.norm(b2_coor, axis = 1)[:, None] # b2 = b1 x CN
    b3_coor = np.cross(b1_coor, b2_coor); b3_coor = b3_coor / np.linalg.norm(b3_coor, axis = 1)[:, None] # b3 = b1 x b2 = CN for a perfect tetrahedron
       
    P_list = np.concatenate([b1_coor, b2_coor, b3_coor], axis = 1)
    P_list = np.reshape(P_list, [np.shape(P_list)[0]*3, 3]).T
    P_list = [P_list[:,i:(i+3)] for i in 3*np.arange(np.shape(b1_coor)[0])]
    P_list = np.linalg.inv(P_list) # Change-of-basis matrix for each position
    
    positions = df.loc[ ((df.Atom =='CB') & (df.Residue!='GLY')) | ((df.Atom =='CA') & (df.Residue=='GLY')), ['coor_x','coor_y','coor_z']]
   
    pos_pairs = np.array(list(itertools.combinations(range(L), 2)))
    P_list_pairs = [P_list[i] for i in pos_pairs[:,0]]
    positions_pairs = positions.to_numpy()[pos_pairs[:,1],:] - positions.to_numpy()[pos_pairs[:,0],:]
    or1_pairs = b1_coor[pos_pairs[:,1],:]
    or2_pairs = b3_coor[pos_pairs[:,1],:]
    
    relative_pairwise_positions = np.einsum('ij,ikj->ik',positions_pairs, P_list_pairs)
    relative_pairwise_or1 = np.einsum('ij,ikj->ik', or1_pairs, P_list_pairs)
    relative_pairwise_or2 = np.einsum('ij,ikj->ik', or2_pairs, P_list_pairs)
        
    aa_seq = df.Residue[df.Atom == 'CA'].to_numpy()
    d = {item: idx for idx, item in enumerate(aa_list)}
    aa_index = np.array([d.get(item) for item in aa_seq])
    aa_pairs = np.concatenate([aa_index[pos_pairs[:,0]][:,None],aa_index[pos_pairs[:,1]][:, None]], axis = 1)
    positions_and_frames = np.concatenate([relative_pairwise_positions, relative_pairwise_or1,
                                           relative_pairwise_or2, aa_pairs], axis = 1)        
    
    dihedrals.Residue = aa_index
    
    return positions_and_frames, dihedrals


# The function `define_frames` parallelizes `get_coordinates` across the entire set of conformation of a given protein ensemble, which can be given either as a folder of .pdb files (one per conformation) or a .xtc and a .top file containing the information for the whole ensemble. The function returns a pair of arrays:
# 
# * The sample of the ensemble's (random) global structure, given as an array of shape [Number of conformations, Number of position pairs, Number of covariates = 11], to which one can access to have conformation/relative position-specific information. For the k-th conformation (with k = 0,1,...) and the positions i,j, the array's element [k,s,:], where s is given by 
#     ```
#     L = 10 # Sequence length
#     pos_pairs = list(itertools.combinations(range(L), 2)) # Pairwise positions 
#     s = pos_pairs.index(tuple((i,j)))
#     ```
#     contains a vector of covariates `[x,y,z,e1_x,e1_y,e1_z,e3_x,e3_y,e3_z,AA_i,AA_j]`, where `x,y,z` are the three coordinates of j-th position in i-th reference system, `e1_x,e1_y,e1_z` (resp. `e3_x, e3_y, e3_z`) are the three coordinates of the j-th first (resp. third) basis vector in the i-th reference system, and `AA_i`, `AA_j` are the identities of the i-th and j-th reside (given as their positions in the alphabetically ordered amino acid list).  
# 
# 
# * The sample of the ensemble's (random) local structure, given as an array of shape [Number of conformations, Sequence length, Number of covariates = 4]. For the k-th conformation (k = 0,1,...) and the i-th sequence position (i = 1,..,L), the array's element [k,i-1,:] contains a vector of covariates `[AA_i, i, phi, psi]`, where `AA_i` is the identity of the residue at position `i` (given as an index in the alphabetically ordered amino acid list), and `phi, psi` are the corresponding values of dihedral angles. 
# 
# Each array can be saved to a .hdf5 file (recommended). .hdf5 files allow accessing to sub-arrays without loading on memory the entire array. This is needed for future computation of ensemble comparison matrices.
# 
# The arguments of `define_frames` are:
# 
# * `pdb_folder`: If the ensemble is given as a folder of .pdb files (one per conformation), the path to such folder. Otherwise, to be set to `None`.
# 
# * `xtc_file`: If the ensemble is given as a (.xtc, .top) file, the path to the .xtc file. Otherwise, to be set to `None`.
# 
# * `top_file`: If the ensemble is given as a (.xtc, .top) file, the path to the .top file. Otherwise, to be set to `None`.
# 
# * `num_cores`: The number of cores to use in parallel computation across the set of conformations.
# 
# * `prot_name`: If the pair of arrays are being saved to a pair of .hdf5 files, the name of the ensemble (string) to name the files. Otherwise, to be set to `None`.
# 
# * `save_to`: The path where the pair of .hdf5 files needs to be saved. If arrays are not saved, to be set to `None`.
# 

# In[ ]:


def define_frames(pdb_folder = None, xtc_file = None, top_file = None, num_cores = 1, prot_name = None, save_to = None, name_variable = '__main__'):
        
    if xtc_file is None and top_file is None and pdb_folder is not None:
        
        traj_file = None
        conf_list = os.listdir(pdb_folder)
        N_conformations = len(conf_list) # Number of conformations
        md_file = md.load_pdb("/".join([pdb_folder,conf_list[0]]))
        L = md_file.topology.n_residues
        N_pairs = len(list(itertools.combinations(range(L), 2)))
        
    elif xtc_file is not None and top_file is not None and pdb_folder is None:
              
        traj_file = md.load_xtc(xtc_file, top = top_file)
        N_conformations = len(traj_file)
        conf_list = np.arange(N_conformations)
        L = traj_file.topology.n_residues
        N_pairs = len(list(itertools.combinations(range(L), 2)))
        
    else:
        quit('Please set pdb_folder != None and xtc_file = top_file = None, or pdb_folder = None and xtc_file != None, top_file != None.')
   
    it_function = partial(get_coordinates, pdb = pdb_folder, traj = traj_file)
       
    def it_function_error(conf):
        
        try:
            output = it_function(conf)
        except:
            output = tuple([np.nan, np.nan])
        return output
    
    # Returning array without saving .hdf5 file
    if prot_name is None and save_to is None:
        
        if __name__ == name_variable:
            
            processed_list = Parallel(n_jobs = num_cores, backend = 'threading')(delayed(it_function_error)(i) for i in tqdm(conf_list))   
   
        ensemble_frames = np.array([l[0] for l in processed_list if np.isnan(l[0]).any() == False])
        ensemble_dihedrals = np.array([l[1] for l in processed_list if np.sum(np.sum(pd.isnull(l[1]))) == 2])
        
        return ensemble_frames, ensemble_dihedrals

    # Saving array into a .hdf5 file (saving one slice per iteration to avoid memory problems)
    elif prot_name is not None and save_to is not None:
        
        # Create and open .hdf5 files
        os.chdir(save_to)
           
        hf_1 = h5py.File("_".join([prot_name,'coordinates.hdf5']), 'w')
        hf_1.create_dataset("ensemble", shape = [N_conformations, N_pairs, 11])
        
        hf_2 = h5py.File("_".join([prot_name,'dihedrals.hdf5']), 'w')        
        hf_2.create_dataset("ensemble", shape = [N_conformations, L, 4])
       
        def it_function_error_saveit(conf):
          
            output = it_function_error(conf)
           
            if np.isnan(output[0]).any() == False:
                hf_1["ensemble"][list(conf_list).index(conf),:,:] = output[0].astype(np.float64)
            else:
                hf_1["ensemble"][list(conf_list).index(conf),:,:] = np.nan
                
            if np.sum(np.sum(pd.isnull(output[1]))) == 2:
                hf_2["ensemble"][list(conf_list).index(conf),:,:] = output[1].to_numpy().astype(np.float64)
            else:
                hf_2["ensemble"][list(conf_list).index(conf),:,:] = np.nan
        
        if __name__ == name_variable:
            
            
            Parallel(n_jobs = num_cores, backend = 'threading')(delayed(it_function_error_saveit)(i) for i in tqdm(conf_list))   
            hf_1.close()
            hf_2.close()     
            
    else:
        quit('Please set prot_name = None and save_to = None or prot_name != None and save_to != None.')


# ## Executing the function
# 
# ### From a (.xtc, .pdb) file

# In[ ]:


'''
# The ensemble is given by a .xtc and a .top file containing the trajectory

xtc_path = "/path_to_xtc_file/xtc_file.xtc" 
top_path = "/path_to_top_file/top_file.top"     
    
n_cores = 1 # Number of cores for parallel computing
ensemble_name = 'my_ensemble' # Name for file saving
save_path = "/coordinates_path" # Folder where arrays will be saved

# Lauch parallel computing (can be time consuming depending on the ensemble size (sequence length, number of conformations) )
define_frames(xtc_file = xtc_path, top_file = top_path, num_cores = n_cores, prot_name = ensemble_name, save_to =  save_path)
'''


# ### From a folder of .pdb files

# In[ ]:


'''
# The ensemble is given by a folder contaning one .pdb file per conformation

pdb_path = "/path_to_pdb_folder" 
n_cores = 1 # Number of cores for parallel computing
ensemble_name = 'my_ensemble' # Name for file saving
save_path = "/coordinates_path" # Folder where arrays will be saved

# Lauch parallel computing (can be time consuming depending on the ensemble size (sequence length, number of conformations) )
define_frames(pdb_folder = pdb_path, num_cores = n_cores, prot_name = ensemble_name, save_to =  save_path)
'''

