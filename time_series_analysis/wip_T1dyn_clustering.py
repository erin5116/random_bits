# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:22:29 2024

test out clustering on OE-MRI data

@author: ywan3672
"""

import SimpleITK as sitk
import os
import glob
import numpy as np
import pylab as plt

#%%
from sklearn.model_selection import train_test_split

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm

#%%Set working directory

# data = '/Users/cbri3325/Desktop/MANGO_DATA/NEW_PROTOCOL/PATIENTS/'
data = 'C:/temp/MANGO_DATA/'
# data = '/mnt/c/temp/MANGO_DATA/'

subjs_path = [ f.path for f in os.scandir(data) if f.is_dir() ] #Create a list of the paths to the subjects directories
subjs_name = [ f.name for f in os.scandir(data) if f.is_dir() ] #Create a list of subjects names
# subjs_name.remove('MANGO_PHANTOM_JUNE')
# subjs_name = ['PATIENT001']

#%%Create a for loop to perform image analysis on each subject sequentially

for current in subjs_name:

    subj_dir = data+current+'/MRI/NIFTI/' #Set path to subject directory
    
    #%%Generate output and figures folders
    
    if not os.path.exists(subj_dir+'outputs/'):#if it does not already exist, create a directory where the optimized dose will be saved
        os.mkdir(subj_dir+'outputs/')
    out_dir = subj_dir+'outputs/'
    
    # if not os.path.exists(subj_dir+'figures/'):#if it does not already exist, create a directory where the optimized dose will be saved
    #     os.mkdir(subj_dir+'figures/')
    # fig_dir = subj_dir+'figures/'

    #### ssign graspvibe sequence, T1 map pre-oxy and T1 map post-oxy paths
    for image in glob.glob(subj_dir+'*t1_dyngraspvibe*'):
        # if 't1map_pre_registered' in image:
        #     pre_t1_path = image
        # elif 't1map_pre_bet' in image:
        #     pre_BET_T1_path = image
        # elif 'post' in image:
        #     post_t1_path = image
        if 't1_dyngraspvibe_1mmiso.nii' in image:
            graspvibe_path = image
        elif 't1_dyngraspvibe_1mmiso_static.nii' in image:
            graspvibe3D_path = image
            
    t1_maps = []
    for image in glob.glob(subj_dir+'*t1_images.*'):
        t1_maps.append(image)
 
    tpts = []
    for image in t1_maps:
        head, tail = os.path.split(image)
        tpts.append(tail[0:5])
    tpts = np.array(tpts)
    
    post = str(max(tpts))
    
    for image in glob.glob(subj_dir+'*t1_images.*'):
        if post in image:
            post_t1_path = image
            
    pre_t1_path = out_dir+'rpre_T1map.nii'
    pre_BET_T1_path = out_dir+'rpre_T1map_bet.nii'