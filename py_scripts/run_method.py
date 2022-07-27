#!/usr/bin/env python
# coding: utf-8

# In[5]:


from build_frames import *
from sample_independent_replicas import *
from multiframe_conversion import *
from wmatrix import *
from wvector import *
from graphical_representation import *

import os
import time
import sys
import shutil
import warnings # Optional
import argparse 

parser = argparse.ArgumentParser(description='Run this method')
parser.add_argument('--n1', help='Name ensamble 1')
parser.add_argument('--n2', help='Name ensamble 2')
parser.add_argument('--results', help='Name results directory')
args=parser.parse_args()

name_1 = args.n1
name_2 = args.n2
results = args.results

Nrep_1 = 1 # If 1, no replicas will be sampled. If multiple replicas are provided by the user, forced to 1.
Nrep_2 = 1
ncores = 'max' # Number of threads (cores). If 'max', set to the maximum number of available threads.


def comparison_tool(ensemble_1_name, ensemble_2_name, results_path = None, interactive = True, N_replicas_1 = 1, N_replicas_2 = 1, N_cores = 1):
    path_to_running = os.path.dirname(os.path.abspath(__file__))
    print(path_to_running)
    ensemble_1_path = os.path.join(path_to_running,'ensemble_1')
    ensemble_2_path = os.path.join(path_to_running,'ensemble_2')
    print(ensemble_1_path)
    print(ensemble_2_path)
    if results_path == ensemble_1_path or results_path == ensemble_2_path:
        sys.exit("Please choose results_path different from ensemble_1_path and ensemble_2_path.")
    if ensemble_1_path == ensemble_2_path:
        sys.exit("Please place each ensemble in one different folder.")
    if results_path is None and not os.path.exists("/".join([os.path.abspath(os.path.join(ensemble_1_path, os.pardir)),"_".join(['results',ensemble_1_name,ensemble_2_name])])):
        os.mkdir("/".join([os.path.abspath(os.path.join(ensemble_1_path, os.pardir)),"_".join(['results',ensemble_1_name,ensemble_2_name])]))
        results_path = "/".join([os.path.abspath(os.path.join(ensemble_1_path, os.pardir)),"_".join(['results',ensemble_1_name,ensemble_2_name])])
    if results_path is None and os.path.exists("/".join([os.path.abspath(os.path.join(ensemble_1_path, os.pardir)),"_".join(['results',ensemble_1_name,ensemble_2_name])])):
        if len(os. listdir("/".join([os.path.abspath(os.path.join(ensemble_1_path, os.pardir)),"_".join(['results',ensemble_1_name,ensemble_2_name])]))) == 0:
            results_path = "/".join([os.path.abspath(os.path.join(ensemble_1_path, os.pardir)),"_".join(['results',ensemble_1_name,ensemble_2_name])])
        else:
            sys.exit("".join(['The folder ', "/".join([os.path.abspath(os.path.join(ensemble_1_path, os.pardir)),"_".join(['results',ensemble_1_name,ensemble_2_name])]),' already exists and it is not empty. Please empty or delete it.']))
    if results_path is not None and not os.path.exists(os.path.join(path_to_running,results_path)):
        results_path = os.path.join(path_to_running,results_path)
        os.mkdir(os.path.join(path_to_running,results_path))
    if results_path is not None and os.path.exists(os.path.join(path_to_running,results_path)):
        if len(os. listdir(results_path))==0:
            results_path = os.path.join(path_to_running,results_path)
        else:
            sys.exit("".join(['The folder ', results_path,' already exists and it is not empty. Please empty or delete it.']))
        
    # Initial parameters
    var_dict = {'multiframe' : 'n', 'check_folder' : True, 'do_xtc_1' : False, 'N_rep_1' : int(N_replicas_1), 'ignore_uncertainty_1' : False, 'do_pdb_1' : False,
                'do_xtc_2' : False,  'N_rep_2' : int(N_replicas_2), 'ignore_uncertainty_2' : False, 'do_pdb_2' : False, 'N1' : 1, 'N2' : 1,
                'ensemble_1_name' : ensemble_1_name, 'ensemble_2_name' : ensemble_2_name, 'ensemble_1_path' : ensemble_1_path, 'ensemble_2_path' : ensemble_2_path}
    
    var_dict['xtc_files_1'] = [file for file in os.listdir(ensemble_1_path)  if file.endswith(".xtc")] 
    var_dict['pdb_files_1'] = [file for file in os.listdir(ensemble_1_path)  if file.endswith(".pdb")]
    var_dict['folders_1'] = [file for file in os.listdir(ensemble_1_path)  if os.path.isdir("/".join([ensemble_1_path,file]))]
    
    var_dict['xtc_files_2'] = [file for file in os.listdir(ensemble_2_path)  if file.endswith(".xtc")]
    var_dict['pdb_files_2'] = [file for file in os.listdir(ensemble_2_path)  if file.endswith(".pdb")]
    var_dict['folders_2'] = [file for file in os.listdir(ensemble_2_path)  if os.path.isdir("/".join([ensemble_2_path,file]))]
    
    print("\n----------------------------------------------------------------------------------\n")
    print(' \(·_·)                                                                  \(·_·)')
    print('   ) )z      Welcome to the beta version of this IDP comparison tool       ) )z')
    print("   / \\                                                                     / \\ \n")
    if interactive == True:
        print("Before launching the computation, let me check I understood everything correctly...")
    print("\n----------------------------------------------------------------------------------\n")
    
    # File processing
    
    for which_ens in ['1','2']:
        
        print("".join(["For the ensemble named ",var_dict["_".join(['ensemble',which_ens,'name'])],', I found ',
                       str(len(var_dict["_".join(['xtc_files',which_ens])])),' .xtc file(s), ',str(len(var_dict["_".join(['pdb_files',which_ens])])),' .pdb file(s) and ',
                       str(len(var_dict["_".join(['folders',which_ens])])),' folder(s).']))
        
        if len(var_dict["_".join(['xtc_files',which_ens])]) + len(var_dict["_".join(['folders',which_ens])]) + len(var_dict["_".join(['pdb_files',which_ens])]) == 0:
            sys.exit("".join(['Folder for ', var_dict["_".join(['ensemble',which_ens,'name'])], ' ensemble is empty...']))
        
        # .xtc files with a .pdb topology file
    
        if len(var_dict["_".join(['xtc_files',which_ens])]) >= len(var_dict["_".join(['pdb_files',which_ens])]) and len(var_dict["_".join(['pdb_files',which_ens])]) == 1:
            if interactive == True:
                print('\nShould I interprete this input as:\n')
            else:
                print('\nI will interprete this input as:\n')
            for j in range(len(var_dict["_".join(['xtc_files',which_ens])])):
                print("".join([str(var_dict["_".join(['xtc_files',which_ens])][j]),' : ',str(j+1),'-th independent replica of ',var_dict["_".join(['ensemble',which_ens,'name'])],',']))
            print("".join([str(var_dict["_".join(['pdb_files',which_ens])][0]),' : topology file for all ',var_dict["_".join(['ensemble',which_ens,'name'])],' replicas.']))
            if interactive == True:
                ens_input = input('...? (y/n)')
            else:
                ens_input = 'y'
            if ens_input == 'n':
                var_dict['multiframe'] = input("Should I ignore .xtc files and consider the .pdb file as a multiframe file? (y/n)")
            else:
                var_dict["_".join(['do_xtc',which_ens])] = True
                var_dict["_".join(['xtc_root_path',which_ens])] = var_dict["_".join(['ensemble',which_ens,'path'])]
                var_dict['check_folder'] = False
                
        # multiframe .pdb files
   
        if var_dict['multiframe'] == 'y' or (len(var_dict["_".join(['pdb_files',which_ens])]) >= 1 and len(var_dict["_".join(['xtc_files',which_ens])]) == 0):
            if interactive == True:
                print('\nShould I interprete this input as:\n')
            else:
                print('\nI will interprete this input as:\n')
            for j in range(len(var_dict["_".join(['pdb_files',which_ens])])):
                if j < len(var_dict["_".join(['pdb_files',which_ens])])-1:
                    end = ','
                else:
                    end = '.'
                print("".join([str(var_dict["_".join(['pdb_files',which_ens])][j]),' : ',str(j+1),'-th independent replica of ',var_dict["_".join(['ensemble',which_ens,'name'])],end]))
            if interactive == True:
                ens_input = input('...? (y/n)')
            else: 
                ens_input = 'y'
            if ens_input == 'y':
                print('Replicas have been given as multiframe .pdb files, which are not supported.')
                print("Converting files to .xtc + topology .pdb...\n ")
                if not os.path.exists("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'converted_files'])):
                    os.mkdir("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'converted_files']))

                for file_j in var_dict["_".join(['pdb_files',which_ens])]:
                    if not os.path.exists("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'converted_files',file_j.split('.pdb')[0]+'.xtc'])):
                        print('Converted file does not exist, converting')
                        multiframe_pdb_to_xtc(pdb_file = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],file_j]), save_path = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'converted_files']), prot_name = file_j.split('.pdb')[0])
                    else:
                        print('Converted file exists, skipping')
                    print("".join(['Done for ',file_j]))
                var_dict["_".join(['do_xtc',which_ens])] = True
                var_dict["_".join(['xtc_root_path',which_ens])] = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'converted_files'])
                var_dict["_".join(['xtc_files',which_ens])] = [file for file in os.listdir(var_dict["_".join(['xtc_root_path',which_ens])]) if file.endswith(".xtc")]
                var_dict["_".join(['pdb_files',which_ens])] = [file for file in os.listdir(var_dict["_".join(['xtc_root_path',which_ens])]) if file.endswith(".pdb")]
                var_dict['check_folder'] = False
                
        # folder with .pdb files
     
        if len(var_dict["_".join(['folders',which_ens])]) >= 1 and var_dict['check_folder'] == True:
            if interactive == True:
                print('\nShould I interprete this input as:\n')
            else:
                print('\nI will interprete this input as:\n')
            for j in range(len(var_dict["_".join(['folders',which_ens])])):
                if j < len(var_dict["_".join(['folders',which_ens])])-1:
                    end = ','
                else:
                    end = '.'
                print("".join([var_dict["_".join(['folders',which_ens])][j],' folder contains: ',str(j+1),'-th independent replica of ',var_dict["_".join(['ensemble',which_ens,'name'])],end]))
            if interactive == True:
                ens_input = input('...? (y/n)')
            else:
                ens_input = 'y'
            if ens_input == 'y':
                var_dict["_".join(['do_pdb',which_ens])] = True
    
        if var_dict["_".join(['do_pdb',which_ens])] == False and var_dict["_".join(['do_xtc',which_ens])] == False:
            sys.exit("".join(['\n Sorry, I did not understood the input. Please follow the guidelines described in the function documentation to create ',eval("_".join(['ensemble',which_ens,'name'])),' folder.\n']))    
            
        print("\n----------------------------------------------------------------------------------\n")
            
        # Sample independent replicas if needed (this will be done after building frames)
        
        if (interactive == True and len(var_dict["_".join(['xtc_files',which_ens])]) == 1 and var_dict["_".join(['do_xtc',which_ens])] == True) or (interactive == True and len(var_dict["_".join(['folders',which_ens])]) == 1 and var_dict["_".join(['do_pdb',which_ens])] == True):
            print("".join(['Only one replica is available for ensemble ',var_dict["_".join(['ensemble',which_ens,'name'])],'.']))
            print("It is possible do extract independent replicas by subsampling from the available one.")
            print("This may not be appropiate if the ensemble corresponds to a MD simulation.")
            subsampling = input("Should I extract independent replicas? (y/n)")
            if subsampling == 'y':
                N_rep = input('Ok. Please take into account the number of conformations and choose how many independent replicas should I extract: (integer)')
                if int(N_rep) <= 0:
                    sys.exit('The number of replicas to sample must be a positive integer.')
                var_dict["_".join(['N_rep',which_ens])] = int(N_rep)
                print("".join(["After computing reference systems, I will extract ",str(N_rep), ' indepedent replicas for ensemble ',var_dict["_".join(['ensemble',which_ens,'name'])],'.\n']))
                print("\n----------------------------------------------------------------------------------\n")
            else:
                var_dict["_".join(['ignore_uncertainty',which_ens])]  = True
                print("\n----------------------------------------------------------------------------------\n")
        
               
        if interactive == False and var_dict["_".join(['N_rep',which_ens])] == 1:
            print("".join(['\nNo independent replicas are being sampled for ensemble ',var_dict["_".join(['ensemble',which_ens,'name'])],'.\n']))
            if (len(var_dict["_".join(['xtc_files',which_ens])]) == 1 and var_dict["_".join(['do_xtc',which_ens])] == True) or (len(var_dict["_".join(['folders',which_ens])]) == 1 and var_dict["_".join(['do_pdb',which_ens])] == True):
                var_dict["_".join(['ignore_uncertainty',which_ens])]  = True        
            
        if interactive == False and var_dict["_".join(['N_rep',which_ens])] > 1:
            print("".join(['\n',str(var_dict["_".join(['N_rep',which_ens])]),' replicas will be sampled for ensemble ',var_dict["_".join(['ensemble',which_ens,'name'])],'.\n']))
            if (len(var_dict["_".join(['xtc_files',which_ens])]) > 1 and var_dict["_".join(['do_xtc',which_ens])] == True) or (len(var_dict["_".join(['folders',which_ens])]) > 1 and var_dict["_".join(['do_pdb',which_ens])] == True):
                print("".join(["More than one replica is already available for ensemble ",var_dict["_".join(['ensemble',which_ens,'name'])],'. Sampling is not possible.\n']))
                var_dict["_".join(['N_rep',which_ens])] = 1             
           
            
    if interactive == True:
        print("Everything seems OK!\n")
        print("".join(['There are ',str(os.cpu_count()),' threads (cores) available.']))
        n_cores = int(input("Please specify the number of threads (cores) you would like to use (positive integer):"))
    else:
        if N_cores == 'max':
            n_cores = int(os.cpu_count())
        else:
            n_cores = int(N_cores)
    
    print("\n----------------------------------------------------------------------------------\n")
    print("3..."); time.sleep(1); 
    print("2..."); time.sleep(1)
    print("1..."); time.sleep(1)
    print("Go!"); time.sleep(0.2)
    print("\n----------------------------------------------------------------------------------")
    
    # Build frames and save coordinates
    
    for which_ens in ['1','2']:
        
        print('\nBuilding reference frames for ' + var_dict["_".join(['ensemble',which_ens,'name'])] + '...\n')
        
        if not os.path.exists("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates'])):
            os.mkdir("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates']))
      
        if var_dict["_".join(['do_xtc',which_ens])] == True:
            
            for j in range(len(var_dict["_".join(['xtc_files',which_ens])])):
                
                if int(var_dict["_".join(['N_rep',which_ens])]) == 1:
                    pname = var_dict["_".join(['ensemble',which_ens,'name'])] + '_' + str(j)
                else:
                    pname = var_dict["_".join(['ensemble',which_ens,'name'])]
                
                print('\nComputing for ' + str(j+1) + '-th replica...\n'); time.sleep(0.5)
                
                define_frames(xtc_file = "/".join([var_dict["_".join(['xtc_root_path',which_ens])],var_dict["_".join(['xtc_files',which_ens])][j]]), top_file = "/".join([var_dict["_".join(['xtc_root_path',which_ens])],var_dict["_".join(['pdb_files',which_ens])][0]]),
                          pdb_folder = None, num_cores = n_cores, prot_name = pname, save_to =  "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates']),
                             name_variable = 'build_frames')
            
        if var_dict["_".join(['do_pdb',which_ens])] == True:
            
            for j in range(len(var_dict["_".join(['folders',which_ens])])):
                
                if int(var_dict["_".join(['N_rep',which_ens])]) == 1:
                    pname = var_dict["_".join(['ensemble',which_ens,'name'])] + '_' + str(j)
                else:
                    pname = var_dict["_".join(['ensemble',which_ens,'name'])]
                
                print('\n Computing for ' + str(j+1) + '-th replica...\n')
                define_frames(xtc_file = None, top_file = None,
                          pdb_folder = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],var_dict["_".join(['folders',which_ens])][j]]), num_cores = n_cores,
                              prot_name = pname, save_to =  "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates']),
                              name_variable = 'build_frames')
                
        # Sample independent replicas if needed
        
        if int(var_dict["_".join(['N_rep',which_ens])]) > 1:
            
            if not os.path.exists("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates_ind_replicas'])):
                os.mkdir("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates_ind_replicas']))
            
            sample_ind_replicas(prot_name = var_dict["_".join(['ensemble',which_ens,'name'])], coordinates_path = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates']), 
                                dihedrals_path = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates']),
                                save_to = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates_ind_replicas']),
                                N_replicas = int(var_dict["_".join(['N_rep',which_ens])]))
    
        # Collect computed frames
        
        if os.path.exists("/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates_ind_replicas'])):
            coor_path = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates_ind_replicas'])
            var_dict["_".join(['coor_path',which_ens])] = coor_path
        else: 
            coor_path = "/".join([var_dict["_".join(['ensemble',which_ens,'path'])],'coordinates'])
        
        var_dict["_".join(['list_global_replicas',which_ens])] = sorted([file for file in os.listdir(coor_path) if file.endswith('coordinates.hdf5')])
        var_dict["_".join(['list_local_replicas',which_ens])] = sorted([file for file in os.listdir(coor_path) if file.endswith('dihedrals.hdf5')])
        
        if len(var_dict["_".join(['list_global_replicas',which_ens])])!=len(var_dict["_".join(['list_local_replicas',which_ens])]):
            print('An error ocurred during frames computation. The number of coordinates and dihedrals files must be the same.')
            print('Computation proceeds by taking the minimum number of replicas.')
            NR = min(len(var_dict["_".join(['list_global_replicas',which_ens])]),len(var_dict["_".join(['list_local_replicas',which_ens])]))
        else:
            NR = len(var_dict["_".join(['list_global_replicas',which_ens])])
        
        if NR == 1:
                 var_dict["_".join(['ignore_uncertainty',which_ens])] = True
        
        var_dict['N' + which_ens] = int(NR)
        
        # Compute intra - ensemble differences
        
        if not os.path.exists("/".join([results_path,'intra_ensemble_wmatrices'])):
            os.mkdir("/".join([results_path,'intra_ensemble_wmatrices']))
        if not os.path.exists("/".join([results_path,'intra_ensemble_wvectors'])):
            os.mkdir("/".join([results_path,'intra_ensemble_wvectors']))
        if not os.path.exists("/".join([results_path,'all_coordinates'])):
            os.mkdir("/".join([results_path,'all_coordinates']))
        
        if var_dict["_".join(['ignore_uncertainty',which_ens])] == False:
                
            for j in range(1,NR):
                
                wmat = w_matrix(prot_1 = var_dict["_".join(['list_global_replicas',which_ens])][0].split('_coordinates.hdf5')[0], prot_2 = var_dict["_".join(['list_global_replicas',which_ens])][j].split('_coordinates.hdf5')[0] , N_centers = 2000, N_cores = n_cores, data_path = coor_path, name_variable = 'wmatrix')
                os.chdir("/".join([results_path,'intra_ensemble_wmatrices']))
                np.save("_".join([var_dict["_".join(['ensemble',which_ens,'name'])],'0',str(j),'wmatrix.npy']), wmat)
            
                wvec = w_vector(prot_1 = var_dict["_".join(['list_local_replicas',which_ens])][0].split('_dihedrals.hdf5')[0], prot_2 = var_dict["_".join(['list_local_replicas',which_ens])][j].split('_dihedrals.hdf5')[0] , N_centers = 2000, N_cores = n_cores, data_path = coor_path, name_variable = 'wvector')
                os.chdir("/".join([results_path,'intra_ensemble_wvectors']))
                np.save("_".join([var_dict["_".join(['ensemble',which_ens,'name'])],'0',str(j),'wvector.npy']), wvec)
            
        for file in os.listdir(coor_path):
            shutil.move("/".join([coor_path,file]),"/".join([results_path,'all_coordinates']))
        os.rmdir(coor_path)            
    
    # Compute inter - ensemble differences
    
    coor_path = "/".join([results_path,'all_coordinates'])
    
    m = np.min([var_dict['N1'], var_dict['N2']])
    a = np.arange(var_dict['N1']); b = np.arange(var_dict['N2'])
    l = [(i,j) for i in a for j in b if i!=j]
    pairs = [(i,i) for i in range(m)]
    
    # The combinations are arbitrary and the user can change the code to choose specific pairs.
    if len(a) > len(b): 
        for k in range(m,len(a)):
            l = [(a[k],j) for j in b]
            pairs.append(l[int(np.random.choice(np.arange(len(l)), 1)[0])])
    
    if len(a) < len(b):
        for k in range(m,len(b)):
            l = [(j, b[k]) for j in a]
            pairs.append(l[int(np.random.choice(np.arange(len(l)), 1)[0])])
    
    if not os.path.exists("/".join([results_path,'inter_ensemble_wmatrices'])):
        os.mkdir("/".join([results_path,'inter_ensemble_wmatrices']))
    
    if not os.path.exists("/".join([results_path,'inter_ensemble_wvectors'])):
        os.mkdir("/".join([results_path,'inter_ensemble_wvectors']))
    
    for j in range(len(pairs)):
        
        wmat = w_matrix(prot_1 = var_dict['list_global_replicas_1'][pairs[j][0]].split('_coordinates.hdf5')[0], prot_2 = var_dict['list_global_replicas_2'][pairs[j][1]].split('_coordinates.hdf5')[0] , N_centers = 2000, N_cores = n_cores, data_path = coor_path, name_variable = 'wmatrix')
        os.chdir("/".join([results_path,'inter_ensemble_wmatrices']))
        np.save("_".join([ensemble_1_name,ensemble_2_name,str(j),'wmatrix.npy']), wmat)
            
        wvec = w_vector(prot_1 = var_dict['list_local_replicas_1'][pairs[j][0]].split('_dihedrals.hdf5')[0], prot_2 = var_dict['list_local_replicas_2'][pairs[j][1]].split('_dihedrals.hdf5')[0] , N_centers = 2000, N_cores = n_cores, data_path = coor_path, name_variable = 'wvector')
        os.chdir("/".join([results_path,'inter_ensemble_wvectors']))
        np.save("_".join([ensemble_1_name,ensemble_2_name,str(j),'wvector.npy']), wvec)
    
    if os.path.exists("/".join([var_dict["_".join(['ensemble_1_path'])],'coordinates'])):
        shutil.rmtree("/".join([var_dict["_".join(['ensemble_1_path'])],'coordinates']))
    if os.path.exists("/".join([var_dict["_".join(['ensemble_1_path'])],'converted_files'])):
        shutil.rmtree("/".join([var_dict["_".join(['ensemble_1_path'])],'converted_files']))
    if os.path.exists("/".join([var_dict["_".join(['ensemble_2_path'])],'coordinates'])):
        shutil.rmtree("/".join([var_dict["_".join(['ensemble_2_path'])],'coordinates']))
    if os.path.exists("/".join([var_dict["_".join(['ensemble_2_path'])],'converted_files'])):
        shutil.rmtree("/".join([var_dict["_".join(['ensemble_2_path'])],'converted_files']))
    
    print("\n----------------------------------------------------------------------------------\n")
    print("Computation done! Here is the result, which has been saved as pdf:")
    print("\n----------------------------------------------------------------------------------\n")
    
    # Print the results
    
    if len(os.listdir("/".join([results_path,'intra_ensemble_wmatrices']))) == 0:
        ind_mat = None
    else:
        ind_mat = "/".join([results_path,'intra_ensemble_wmatrices'])
    
    if len(os.listdir("/".join([results_path,'intra_ensemble_wvectors']))) == 0:
        ind_vec = None
    else:
        ind_vec = "/".join([results_path,'intra_ensemble_wvectors'])
    
    wmatrix_plot(prot_name_1 = ensemble_1_name, prot_name_2 = ensemble_2_name,
             wmatrix_path = "/".join([results_path,'inter_ensemble_wmatrices']), 
             wmatrix_ind_folder = ind_mat,
             wvector_path = "/".join([results_path,'inter_ensemble_wvectors']),
             wvector_ind_folder = ind_vec,
             save_path = results_path)




comparison_tool(ensemble_1_name = name_1, 
                ensemble_2_name = name_2, 
                results_path = results,
                interactive = False,
                N_replicas_1 = Nrep_1,
                N_replicas_2 = Nrep_2,
                N_cores = ncores)






