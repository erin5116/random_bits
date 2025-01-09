# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:53:12 2025

To convert colormap files from https://github.com/mfuderer/colorResources to color tables for 3D Slicer

@author: ywan3672
"""

import os
from os.path import join, exists, basename
from glob import glob
import pandas as pd
import csv
import numpy as np

#%%

data_path = r'C:\Users\ywan3672\Downloads\colorResources-main\colorResources-main'

save_path = r'C:\temp\random_bits'

# Choose which colormap to convert
#### For T1 maps
# file_name = 'lipari.csv' # recommended for T1 maps (Fuderer et al 2024)
# _range = (1000,5000) # ms, the range used for T1 map in the SI-BiRT studies
#### For R2* maps
file_name = 'navia.csv' # recommended for T2/R2 maps (Fuderer et al 2024)
_range = (0,80) # s-1, the range used for R2* map in the SI-BiRT studies

alpha = 255
cvt_original = True
cvt_logged = True
#%% functions

def new_ind_log(g,mapLength=256,loLev=0,upLev=4000):
    # adapted from https://github.com/mfuderer/colorResources/blob/main/colorLogRemap.m
    
    # mapLength = len(df)
    eInv = np.exp(-1.0)
    aVal = eInv*upLev
    mVal = np.max([aVal, loLev])
    bVal = (1.0/mapLength) + (aVal-loLev)/(2*aVal-loLev)
    bVal = bVal+0.0000001
    
    logPortion = 1.0 / (np.log(mVal) - np.log(upLev))
    f = 0.0
    x = g*(upLev-loLev)/mapLength+loLev
    
    if x>mVal:
        f = mapLength * ((np.log(mVal) - np.log(x)) * logPortion * (1 - bVal) + bVal)
    else:
        if (loLev<aVal) & (x>loLev):
            f = mapLength * ( (x - loLev) / (aVal-loLev) * (bVal - (1.0/mapLength))) + 1.0
        elif x<= loLev:
            f=1.0
    
    tempInd = int(min([mapLength,np.floor(f)]))
    return tempInd

def color_remap_ind(mapLength,loLev,upLev):
    
    remap_ind = []
    for i in range(mapLength):
        remap_ind.append(new_ind_log(i,mapLength,loLev,upLev))
    
    return remap_ind


#%% load colormap

map_name = file_name.split('.')[0]

file_path = join(data_path,file_name)

map_list = []
with open(file_path, "r") as f:
    reader = csv.reader(f, delimiter="\n")
    for i, line in enumerate(reader):
        temp_line = line[0].split(' ')
        # read each line of R, G, B
        temp_line = [float(i) for i in temp_line]
        # the original file range is 0-1, scale to convert to 0-255
        temp_line = [int(i*255) for i in temp_line]
        map_list.append(temp_line)

df = pd.DataFrame(map_list,columns=['R','G','B'])

#%% convert original colormap to 3D slicer format, using the inbuilt Magma as template

if cvt_original:
    f = open('{0}.ctbl'.format(map_name),'w')
    
    f.write('# Color table file {0}\n'.format(save_path))
    f.write('# {0} values\n'.format(len(df)))
    
    for ind,each_row in df.iterrows():
        
        f.write('{0} {0} {1} {2} {3} {4}\n'.format(ind,each_row['R'],each_row['G'],each_row['B'],alpha))
    
    f.close()

#%% process logarithmic

if cvt_logged:
    
    loLev = _range[0]
    upLev = _range[1]
    
    new_ind = color_remap_ind(len(df),loLev,upLev)
    
    f = open('{0}_logmapped.txt'.format(map_name),'w')
    f.write('# Color procedural file {0}\n'.format(save_path))
    f.write('# {0} values\n'.format(len(set(new_ind))))
    
    for each_ind in new_ind:
        
        map_ind = new_ind.index(each_ind)
        each_row = df.iloc[each_ind]
        
        f.write('{0} {1} {2} {3} {4}\n'.format(map_ind,each_row['R']/255,each_row['G']/255,each_row['B']/255,alpha/255))
    f.close()


