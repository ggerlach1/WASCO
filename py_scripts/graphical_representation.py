#!/usr/bin/env python
# coding: utf-8

# # Graphical representation of global and local differences

# Import required libraries

# In[ ]:


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# The function `wmatrix_plot` returns a graphical representation of both global and local differences between a pair of ensembles, taking uncertainty into account and providing an overall score/distance between the whole ensembles. Its arguments are
# 
# * `prot_name_1`: the name of the first ensemble (to be displayed on the plot).
# * `prot_name_2`: the name of the second ensemble (to be displayed on the plot).
# * `wmatrix_path`: a path to a folder where all the global distance files (generated with the function `w_matrix`) between replicas of different ensembles are stored.
# * `wvector_path`: a path to a folder where all the local distance files (generated with the function `w_vector`) between replicas of different ensembles are stored.
# * `wmatrix_ind_folder`: a path to a folder where all the global distance files (generated with the function `w_matrix`) between replicas of the same ensemble are stored. If set to `None`, uncertainty in global differences is ignored.
# * `wvector_ind_folder`: a path to a folder where all the local distance files (generated with the function `w_vector`) between replicas of the same ensemble are stored. If set to `None`, uncertainty in local differences is ignored.
# * `save_path`: if not `None`, a path where to save in .pdf the produced plot.

# In[2]:


def wmatrix_plot(prot_name_1, prot_name_2, wmatrix_path, wvector_path, wmatrix_ind_folder = None, wvector_ind_folder = None, save_path = None):
         
    # Load Wasserstein vector data (local differences)    
        
    os.chdir(wvector_path)
    vectors_list = list()

    for j in range(len(os.listdir(wvector_path))):
        wvector_j = np.load(os.listdir(wvector_path)[j])
        vectors_list.append(wvector_j)
    
    wvector = vectors_list[0]
    wvector[:,1] = sum(vectors_list)[:,1]/len(vectors_list) # Average of local differences between the pair of ensembles
   
    if wvector_ind_folder is not None: # Adding independent replicas uncertainty
        
        # Independent replicas W-vectors

        os.chdir(wvector_ind_folder)
        ind_vectors_list = list()

        for j in range(len(os.listdir(wvector_ind_folder))):
            wvector_j = np.load(os.listdir(wvector_ind_folder)[j])
            ind_vectors_list.append(wvector_j)
    
        ind_vector = sum(ind_vectors_list)[:,1]/len(ind_vectors_list) # Average of local differences between replicas of a same ensemble
    
    if wvector_ind_folder is None:
        ind_vector = np.zeros(np.shape(wvector[:,1]))
    
    # Local dissimilarity: ignoring uncertainty (metric properties satisfied)
    local_dissimilarity = np.sqrt(np.nansum(wvector[:,1]**2)) 
    
    wvector[:,1] = wvector[:,1] - ind_vector
    wvector[:,1] = wvector[:,1]*(wvector[:,1]>0)   
    pv = np.exp(-8*wvector[:,2]*wvector[:,3]/(wvector[:,2]+wvector[:,3])*wvector[:,1]**2) # p-values
    wvector_sig = wvector[pv < 0.05, : ] 
    
    if wvector_ind_folder is not None:
        wvector_ind_comparison = wvector[:,1] / ind_vector
    else:
        wvector_ind_comparison = wvector[:,1]
        
    # Local dissimilarity: including uncertainty (metric properties not satisfied)
    #local_dissimilarity = np.sqrt(np.nansum(wvector[:,1]**2))
   
    
    # Load Wasserstein matrix data (global differences)
    
    os.chdir(wmatrix_path)
    wmatrices_list = list()
  
    for j in range(len(os.listdir(wmatrix_path))): 
        wmatrix_j = np.load(os.listdir(wmatrix_path)[j])
            
        wmatrix_plot_j = np.zeros([int(np.max(wmatrix_j[:,1]))+1,int(np.max(wmatrix_j[:,1]))+1])
        wmatrix_plot_j[:,:] = np.nan

        for i in range(np.shape(wmatrix_j)[0]):
            wmatrix_plot_j[int(wmatrix_j[i,0]),int(wmatrix_j[i,1])] = wmatrix_j[i,2]
        
        wmatrices_list.append(wmatrix_plot_j)
    
    wmatrix_plot = sum(wmatrices_list)/len(wmatrices_list) # Average of global differences between the pair of ensembles

    if wmatrix_ind_folder is not None: # Adding independent replicas uncertainty
        
        # Independent replicas W-matrices

        os.chdir(wmatrix_ind_folder)
        ind_matrices_list = list()

        for j in range(len(os.listdir(wmatrix_ind_folder))):
            wmatrix_j = np.load(os.listdir(wmatrix_ind_folder)[j])
            
            wmatrix_plot_j = np.zeros([int(np.max(wmatrix_j[:,1]))+1,int(np.max(wmatrix_j[:,1]))+1])
            wmatrix_plot_j[:,:] = np.nan

            for i in range(np.shape(wmatrix_j)[0]):
                wmatrix_plot_j[int(wmatrix_j[i,0]),int(wmatrix_j[i,1])] = wmatrix_j[i,2]
            
            ind_matrices_list.append(wmatrix_plot_j)
    
        ind_matrix = sum(ind_matrices_list)/len(ind_matrices_list) # Average of local differences between replicas of a same ensemble
     
    # Global dissimilarity: ignoring uncertainty (metric properties satisfied)
    overall_dissimilarity = np.sqrt(np.nansum(wmatrix_plot**2))    
        
    if wmatrix_ind_folder is None:
        ind_matrix = np.zeros(np.shape(wmatrix_plot))
    
    wmatrix_plot = wmatrix_plot - ind_matrix
    wmatrix_plot = wmatrix_plot*(wmatrix_plot>0)
    
    if wmatrix_ind_folder is not None:
        wmatrix_ind_comparison = wmatrix_plot / ind_matrix
    else:
        wmatrix_ind_comparison = wmatrix_plot
        
    # Global dissimilarity: including uncertainty (metric properties not satisfied)    
    #overall_dissimilarity_ind = np.sqrt(np.nansum(wmatrix_plot**2))
    
    # Add vector on diagonal
    
    wmatrix_vector = np.zeros(np.shape(wmatrix_plot))
    wmatrix_vector[:] = np.nan
    for k in range(1,np.shape(wmatrix_vector)[0]-1):
        wmatrix_vector[k,k] = wvector_ind_comparison[k-1]
    
    
    # Plot matrix
    
    if wmatrix_ind_folder is not None:
        r3_label = '$R^3 \Delta W$ / $W_{ind}$'
    else:
        r3_label = '$R^3$ Wasserstein distance'
    if wvector_ind_folder is not None:
        torus_label = '$T^2 \Delta W$ / $W_{ind}$'
    else:
        torus_label = '$T^2$ Wasserstein distance'
        
    res = sns.heatmap(wmatrix_ind_comparison.T, cmap='Reds',square=True,  cbar_kws={"shrink": .5,
                                                                        'label': r3_label,
                                                                        'location': 'left'})
    res.figure.axes[-1].yaxis.label.set_size(15)
    plt.suptitle(" ".join([prot_name_1,'-',prot_name_2,'Wasserstein matrix']), fontsize=10)
    plt.title("".join(['Overall global dissimilarity = ',str(np.round(overall_dissimilarity,3)),
                       '\nOverall local dissimilarity = ', str(np.round(local_dissimilarity,3))]), fontsize = 7)
    plt.xlabel('Sequence position')
    plt.ylabel('Sequence position')
    plt.xticks(rotation=0) 
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 5)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 5)
    sns.heatmap(wmatrix_vector.T, cmap='Blues',square=True,  cbar_kws={"shrink": .35,
                                                                    'label': torus_label, 'location' : 'right'})
    res.figure.axes[-1].yaxis.label.set_size(15)
    sns.scatterplot(x = wvector_sig[:,0] + 0.5 , y =  wvector_sig[:,0] + 0.5, color='white',
                    edgecolor='blue', s = 30, marker ='*', legend = False)
    if save_path is not None:
        os.chdir(save_path)
        plt.savefig("".join(['wmatrix_wvector_',prot_name_1,'_',prot_name_2,'.pdf']), bbox_inches='tight') 
    plt.show()
 


# ## Executing the function

# In[ ]:


'''
mat_path = "/path_to_global_differences_folder"
mat_ind_path = "/path_to_independent_replicas_global_differences_folder"

vec_path = "/path_to_local_differences_folder"
vec_ind_path = "/path_to_independent_replicas_local_differences_folder"

prot_name_1 = 'my_ensemble_1' # Names for plot title
prot_name_2 = 'my_ensemble_2'

save_here = "/save_plot_in_pdf_here"

wmatrix_plot(prot_name_1 = prot_name_1, prot_name_2 = prot_name_2,
             wmatrix_path = mat_path, 
             wmatrix_ind_folder = mat_ind_path,
             wvector_path = vec_path,
             wvector_ind_folder = vec_ind_path,
             save_path = save_here)
'''

