#!/usr/bin/env python
# coding: utf-8

# # Compute Wasserstein vector between two ensembles
# ## Difference between the empirical local structures

# Important note: Python version needs to be set to __Python 3.8__ to be able to load the library `faiss`.

# In[ ]:


from joblib import Parallel, delayed
from functools import partial
import os
import numpy as np
import h5py


# The function `wasserstein_i` computes __Wasserstein distance__ between the i-th dihedral angles distributions (the i-th element of the __empirical local structures__) of a pair of (replicas of two) ensembles. This function will be parallelized across all sequence positions. Its arguments are:
# 
# * `which_pos`: an integer (i) specifying the i-th sequence position.
# 
# * `prot_name_1`: the name of the first ensemble, whose .hdf5 dihedrals file is `prot_name_1_dihedrals.hdf5`.
# 
# * `prot_name_2`: the name of the second ensemble, whose .hdf5 dihedrals file is `prot_name_2_dihedrals.hdf5`.
# 
# * `ncenters`: the number of clusters when kmeans clustering needs to be performed.
# 
# * `coor_path`: the path where all the dihedrals .hdf5 files are located.
# 
# Normally, the user can load `wasserstein_i` and skip to the next function `w_vector`, which integrates `wasserstein_i` and its arguments, and should be used if the complete vector for all sequence positions need to be computed.

# In[ ]:


def wasserstein_i(which_pos, prot_name_1, prot_name_2, ncenters, coor_path):
    
    # Required libraries (need to be placed here for parallel computing)
    
    import os
    import ot
    import h5py
    import numpy as np
    import pandas as pd
    import faiss # Needs Python 3.8
    from scipy.spatial.distance import cdist
    
    # Required functions (need to placed here for parallel computing)
    
    def clustering_torus(prot_elements,n_centers): # Performs kmeans for a sample on the two-dimensional flat torus
        
        # The clustering is performed by parameterizing (phi, psi) angles as the elements
        # (cos(phi), sin(phi), cos(psi), sin(psi)) of R^4. This is known as extrinsic kmeans.
        
        prot_elements = np.concatenate([np.cos(prot_elements[:,0])[:,None],
                                        np.sin(prot_elements[:,0][:,None]),
                                        np.cos(prot_elements[:,1][:,None]),
                                        np.sin(prot_elements[:,1][:,None])], axis = 1) # Parameterize (phi,psi) values
        
        kmeans = faiss.Kmeans(d=4, k = ncenters, min_points_per_centroid = 1, max_points_per_centroid = 10000000)
        kmeans.train(prot_elements.astype(np.float32))
        a = np.concatenate([np.arctan2(kmeans.centroids[:,1],kmeans.centroids[:,0])[:,None],
                            np.arctan2(kmeans.centroids[:,3],kmeans.centroids[:,2])[:,None]],axis=1) # Back to (phi,psi) after clustering
       
            
        kmeans_labels = kmeans.index.search(prot_elements.astype(np.float32),1)[1] # Support of the "clustered distribution"
        empty_clusters = np.setdiff1d(np.arange(n_centers), kmeans_labels) # Check for empty clusters
        mass = pd.DataFrame(kmeans_labels).value_counts().sort_index()
        mass_norm = np.array(mass/np.sum(mass)) # Probability mass assigned to each cluster
        
        if len(empty_clusters) > 0: # Remove empty clusters
            a = np.delete(a, empty_clusters, axis = 0)
      
        return a, mass_norm
    
    def metric2_torus(x,y): # Geodesic distance on the two-dimensional flat torus
        
        return sum(np.minimum(np.abs(x-y),1-np.abs(x-y))**2)
    
    ###########################################################################
   
    # Extract i-th (phi, psi) distributions for both ensembles
    os.chdir(coor_path)
    
    h5f_1 = h5py.File("_".join([prot_name_1,'dihedrals.hdf5']),'r')
    L1 = int(0.5*(1 + np.sqrt(1 + 8*np.shape(h5f_1['ensemble'])[1])))
    pos_ij_prot_1 = h5f_1['ensemble'][:, which_pos , :] # Access only to angles of i-th residue
    h5f_1.close()
     
    h5f_2 = h5py.File("_".join([prot_name_2,'dihedrals.hdf5']),'r')
    L2 = int(0.5*(1 + np.sqrt(1 + 8*np.shape(h5f_2['ensemble'])[1])))
    pos_ij_prot_2 = h5f_2['ensemble'][:, which_pos , :] # Access only to angles of i-th residue
    h5f_2.close()
    
    pos_ij_prot_1 = pos_ij_prot_1[~np.isnan(pos_ij_prot_1).any(axis=1),:] # Remove missing conformations
    pos_ij_prot_2 = pos_ij_prot_2[~np.isnan(pos_ij_prot_2).any(axis=1),:] 
       
    if L1 != L2: 
        quit('Both ensembles must have the same length') 
    
    n = np.shape(pos_ij_prot_1)[0] # Number of conformations of the first ensemble
    m = np.shape(pos_ij_prot_2)[0] # Number of conformations of the second ensemble
           
    if n <= ncenters: # If clustering is not needed for the first ensemble
        
        a = pos_ij_prot_1[:,2:4]
        ma = np.ones(n)/n
            
    else: # Clustering for the first ensemble
                        
        a, ma = clustering_torus(pos_ij_prot_1[:,2:4], ncenters) 
    
    if m <= ncenters: # If clustering is not needed for the second ensemble
                        
        b = pos_ij_prot_2[:,2:4]
        mb = np.ones(m)/m
            
    else: # Clustering for the second ensemble
                      
        b, mb = clustering_torus(pos_ij_prot_2[:,2:4], ncenters)
       
    M = cdist(a, b, metric2_torus) # Cost matrix
    clean = ot.utils.clean_zeros(ma, mb, M)
    w_i = np.sqrt(ot.emd2(clean[0], clean[1], clean[2])) # 2-Wasserstein distance    
   
    return w_i, n, m # Returning Wasserstein distance together with sample sizes (needed to compute p-values)
  


# The function `w_vector` parallelizes `wasserstein_i` across the list of all sequence positions. Therefore, it computes the Wasserstein vector representing the difference between the empirical local structures of a pair of (replicas of two) ensembles. The function returns an array of shape [number of sequence positions, 5] array, ready to be graphically represented (using function `plot_matrix`). For a given sequence position i, (i = 1,...,L), the array's element [i-1,:] contains the vector `[i, w_i, n, m, AA_i]`, where `w_i` is the Wasserstein distance between both ensemble's i-th dihedrals distributions, `n`, `m` the number of conformations of the first and second ensemble respectively, and `AA_i` the identity of the i-th residue (given as its index in the alphabetically ordered amino acid list).
# 
# The arguments of `w_vector` are:
# 
# * `prot_name_1`: the name of the first ensemble, whose .hdf5 dihedrals file is `prot_name_1_dihedrals.hdf5`.
# 
# * `prot_name_2`: the name of the second ensemble, whose .hdf5 dihedrals file is `prot_name_2_dihedrals.hdf5`.
# 
# * `N_centers`: the number of clusters when kmeans clustering needs to be performed.
# 
# * `data_path`: the path where all the dihedrals .hdf5 files are located.
# 
# The computation time may be long, depending on both ensemble sizes (sequence length, number of conformations).
# 

# In[ ]:


def w_vector(prot_1, prot_2, N_centers, N_cores, data_path, name_variable = '__main__'):
  
    os.chdir(data_path) # Some data needs to be loaded to extract needed parameters
    h5f_1 = h5py.File("_".join([prot_1,'dihedrals.hdf5']),'r')
    L1 = np.shape(h5f_1['ensemble'])[1] # Sequence length
    it_pos = range(1,L1-1) # List of sequence positions
    res_list = h5f_1['ensemble'][0,1:(L1-1),0][:,None] # List of residues identities along the sequence
    del(h5f_1) # Free memory
    
    it_function = partial(wasserstein_i, prot_name_1 = prot_1, prot_name_2 = prot_2,
                         ncenters = N_centers, coor_path = data_path) 
    
    print('-------------------------------------------------------------------\n')
    print("".join(['Computing Wasserstein distances for ', str(len(res_list)), ' sequence positions.\n']))
    print("".join(['Protein 1 : ', prot_1,'\n']))
    print("".join(['Protein 2 : ', prot_2,'\n']))
    print('-------------------------------------------------------------------\n')   

    
    if __name__ == name_variable: # Parallel computing
        w_distances = Parallel(n_jobs = N_cores, verbose=10, backend = 'threading')(delayed(it_function)(i) for i in it_pos)
    
    positions = np.asarray(it_pos)[:,None]
    distances = np.reshape(np.asarray(w_distances), [np.shape(w_distances)[0], 3])
    
    return np.concatenate([positions, distances, res_list], axis = 1)


# ## Executing the function

# In[ ]:


'''
prot_name_1 = "my_ensemble_1"
prot_name_2 = "my_ensemble_2"
coordinates_path = "/path_to_coordinates_folder" # Folder where my_ensemble_1_dihedrals.hdf5 and my_ensemble_2_dihedrals.hdf5 are located

n_clusters = 2000 # Recommended number of clusters
n_cores = 1 # Number of cores for parallel computing

wvector = w_vector(prot_1 = prot_name_1, prot_2 = prot_name_2 , N_centers = n_clusters, N_cores = n_cores, data_path = coordinates_path)

# The resulting vector should be saved (needed for graphic representation)
os.chdir('save_in_this_path')
np.save("_".join([prot_name_1,prot_name_2,'wvector.npy']), wmatrix)
'''

