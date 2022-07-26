#!/usr/bin/env python
# coding: utf-8

# # Compute Wasserstein matrix between two ensembles
# ## Difference between the empirical global structures

# Important note: Python version needs to be set to __Python 3.8__ to be able to load the library `faiss`.

# In[ ]:


from joblib import Parallel, delayed
from functools import partial
import os
import numpy as np
import itertools
import h5py
import warnings # Optional
warnings.filterwarnings("ignore") # Optional


# The function `wasserstein_ij` computes __Wasserstein distance__ between the relative i,j position distributions (the i,j element of the __empirical global structures__) of a pair of (replicas of two) ensembles. This function will be parallelized across the list of all pairwise positions along the sequence. Its arguments are:
# 
# * `which_pos`: a tuple of two integers `tuple((pos_i, pos_j))`, where `pos_i` and `pos_j` denote the i-th and j-th sequence positions respectively.
# 
# * `prot_name_1`: the name of the first ensemble, whose .hdf5 coordinates file is `prot_name_1_coordinates.hdf5`.
# 
# * `prot_name_2`: the name of the second ensemble, whose .hdf5 coordinates file is `prot_name_2_coordinates.hdf5`.
# 
# * `ncenters`: the number of clusters when kmeans clustering needs to be performed.
# 
# * `coor_path`: the path where all the coordinates .hdf5 files are located.
# 
# Normally, the user can load `wasserstein_ij` and skip to the next function `w_matrix`, which integrates `wasserstein_ij` and its arguments, and should be used if the complete matrix for all pairwise relative positions need to be computed.

# In[ ]:


def wasserstein_ij(which_pos, prot_name_1, prot_name_2, ncenters, coor_path):
    
    # Required libraries (need to be placed here for parallel computing)
   
    import faiss # Needs Python 3.8
    import h5py
    import os
    import ot
    import itertools
    import numpy as np
    import pandas as pd
           
    ########################################
    
    # Extract relative i,j positions distributions for both ensembles
    
    os.chdir(coor_path)
  
    h5f_1 = h5py.File("_".join([prot_name_1,'coordinates.hdf5']),'r')
    L1 = int(0.5*(1 + np.sqrt(1 + 8*np.shape(h5f_1['ensemble'])[1]))) # Sequence length
    pos_pairs_1 = list(itertools.combinations(range(L1), 2)) # List of pairwise positions
    pos_ij_prot_1 = h5f_1['ensemble'][:, pos_pairs_1.index(which_pos), :] # Access to i,j relative position distribution
    h5f_1.close()

    h5f_2 = h5py.File("_".join([prot_name_2,'coordinates.hdf5']),'r')
    L2 = int(0.5*(1 + np.sqrt(1 + 8*np.shape(h5f_2['ensemble'])[1]))) # Sequence length
    pos_pairs_2 = list(itertools.combinations(range(L2), 2)) # List of pairwise positions
    pos_ij_prot_2 = h5f_2['ensemble'][:, pos_pairs_2.index(which_pos), :] # Access to i,j relative position distribution
    h5f_2.close()
    
    pos_ij_prot_1 = pos_ij_prot_1[~np.isnan(pos_ij_prot_1).any(axis=1),:] # Remove missing conformations
    pos_ij_prot_2 = pos_ij_prot_2[~np.isnan(pos_ij_prot_2).any(axis=1),:] 
       
    if L1 != L2: 
        quit('Both ensembles must have the same length') 
    
    n = np.shape(pos_ij_prot_1)[0] # Number of conformations in the first ensemble
    m = np.shape(pos_ij_prot_2)[0] # Number of conformations in the second ensemble
    
    # Keeping (x,y,z) coordinates for clustering in R3
    
    pos_ij_prot_1 = pos_ij_prot_1[:,0:3]
    pos_ij_prot_2 = pos_ij_prot_2[:,0:3]
        
    # Check is clustering is needed
    
    if n <= ncenters: # No clustering for the first ensemble
        
        a = pos_ij_prot_1
        ma = np.ones(n)/n
            
    else: # Clustering for the first ensemble
         
        kmeans_1 = faiss.Kmeans(d = 3, k = ncenters, min_points_per_centroid = 1, max_points_per_centroid = 10000000)
        kmeans_1.train(pos_ij_prot_1.astype(np.float32))
            
        #Check for empty clusters
        kmeans_labels = kmeans_1.index.search(pos_ij_prot_1.astype(np.float32),1)[1]
        empty_clusters = np.setdiff1d(np.arange(ncenters), kmeans_labels)
       
        a = kmeans_1.centroids # Support of the "clustered distribution"
        mass_a = pd.DataFrame(kmeans_labels).value_counts().sort_index() 
        ma = np.array(mass_a/np.sum(mass_a)) # Probability mass assigned to each cluster
            
        if len(empty_clusters) > 0:
            
            a = np.delete(a, empty_clusters, axis = 0) # Remove empty clusters
      
    if m <= ncenters: # No clustering for the second ensemble
        
        b = pos_ij_prot_2
        mb = np.ones(m)/m
    
    else: # Clustering for the second ensemble
               
        kmeans_2 = faiss.Kmeans(d = 3, k = ncenters, min_points_per_centroid = 1, max_points_per_centroid = 10000000)
        kmeans_2.train(pos_ij_prot_2.astype(np.float32))
            
        #Check for empty clusters
        kmeans_labels = kmeans_2.index.search(pos_ij_prot_2.astype(np.float32),1)[1]
        empty_clusters = np.setdiff1d(np.arange(ncenters), kmeans_labels)
       
        b = kmeans_2.centroids # Support of the "clustered distribution"
        mass_b = pd.DataFrame(kmeans_labels).value_counts().sort_index()
        mb = np.array(mass_b/np.sum(mass_b)) # Probability mass assigned to each cluster
            
        if len(empty_clusters) > 0:
            
            b = np.delete(b, empty_clusters, axis = 0) # Remove empty clusters
      
         
    # Compute Wasserstein distance     
         
    M = ot.dist(a, b, metric = 'sqeuclidean') # Cost matrix
    clean = ot.utils.clean_zeros(ma, mb, M)
    w_ij = np.sqrt(ot.emd2(clean[0], clean[1], clean[2])) # 2-Wasserstein distance
        
    return w_ij
  


# The function `w_matrix` parallelizes `wasserstein_ij` across the list of all pairwise relative positions along the sequence. Therefore, it computes the Wasserstein matrix representing the difference between the empirical global structures of a pair of (replicas of two) ensembles. The function returns an array of shape [number of position pairs, 3] array, ready to be graphically represented (using function `plot_matrix`). For a given pair of positions i,j, the array's element [s,:], where s is given by
# 
#     L = 10 # Sequence length
#     pos_pairs = list(itertools.combinations(range(L), 2)) # Pairwise positions 
#     s = pos_pairs.index(tuple((i,j)))
#     
# contains the vector `[pos_i, pos_j, w_ij]`, where `pos_i` and `pos_j` are respectively the i-th and j-th sequence positions (starting form zero), and `w_ij` the computed Wasserstein distance between the i,j relative position distributions of both ensembles.
# 
# 
# The arguments of `w_matrix` are:
# 
# * `prot_name_1`: the name of the first ensemble, whose .hdf5 coordinates file is `prot_name_1_coordinates.hdf5`.
# 
# * `prot_name_2`: the name of the second ensemble, whose .hdf5 coordinates file is `prot_name_2_coordinates.hdf5`.
# 
# * `N_centers`: the number of clusters when kmeans clustering needs to be performed.
# 
# * `data_path`: the path where all the coordinates .hdf5 files are located.
# 
# The computation time may be long, depending on both ensemble sizes (sequence length, number of conformations).
# 

# In[ ]:


def w_matrix(prot_1, prot_2, N_centers, N_cores, data_path, name_variable = '__main__'):
    
    
    # Some data need to be loaded to get basic parameters
    os.chdir(data_path) 
    h5f_1 = h5py.File("_".join([prot_1,'coordinates.hdf5']),'r')
    L1 = int(0.5*(1 + np.sqrt(1 + 8*np.shape(h5f_1['ensemble'])[1]))) # Sequence length
    it_pairs = list(itertools.combinations(range(L1), 2)) # List of all pairwise relative positions
    del(h5f_1) # Clear memory
    
    it_function = partial(wasserstein_ij, prot_name_1 = prot_1, prot_name_2 = prot_2,
                         ncenters = N_centers, coor_path = data_path) 
      
    print('-------------------------------------------------------------------\n')
    print("".join(['Computing pairwise Wasserstein distances for ', str(len(it_pairs)), ' pairs of sequence positions.\n']))
    print("".join(['Protein 1 : ', prot_1,'\n']))
    print("".join(['Protein 2 : ', prot_2,'\n']))
    print('-------------------------------------------------------------------\n')   
    
    if __name__ == name_variable: # Parallel computing
        pairwise_distances = Parallel(n_jobs = N_cores, verbose=10, backend = 'threading')(delayed(it_function)(i) for i in it_pairs)
    
    pairs = np.asarray(it_pairs)
    distances = np.reshape(np.asarray(pairwise_distances), [len(pairwise_distances), 1])
    
    return np.concatenate([pairs, distances], axis = 1)


# ## Executing the function

# In[ ]:


'''
prot_name_1 = "my_ensemble_1"
prot_name_2 = "my_ensemble_2"
coordinates_path = "/path_to_coordinates_folder" # Folder where my_ensemble_1_coordinates.hdf5 and my_ensemble_2_coordinates.hdf5 are located

n_clusters = 2000 # Recommended number of clusters
n_cores = 1 # Number of cores for parallel computing

wmatrix = w_matrix(prot_1 = prot_name_1, prot_2 = prot_name_2 , N_centers = n_clusters, N_cores = n_cores, data_path = coordinates_path)

# The resulting matrix should be saved (needed for graphic representation)
os.chdir('save_in_this_path')
np.save("_".join([prot_name_1,prot_name_2,'wmatrix.npy']), wmatrix)
'''

