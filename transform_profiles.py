# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:50:09 2020
READS PROFILES AND TRANSFORMS AND SAVES TO SEND TO KUL:
    Input: ev profiles with 400+ indep profiles
    output: ev profiles with 300 columns corresponding to each household
@author: U546416
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import EVmodel
import util_occupancy
import os
import util
import time
###########################################################################################

#################                INPUT DATA                                 ###############

###########################################################################################

# CHANGE THIS TO SELECT THE OCCUPANCY PROFILES FOLDER/FILE
folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimFiles\\'
file = 'OccupancyTotal.csv'
folder_base_load = folder + 'Inputs for grid simulations\\'
folder_hp_load = folder + 'Re-ordered profiles\HPprofiles\\'
folder_pv = folder + 'Re-ordered profiles\PVprofiles\\'

# Folder of results
results_folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimResults\\'

# output folder to save
output_folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimResults\To_Send\\'

util.create_folder(output_folder)

###########################################################################################


# Reading pre-computed schedules
schedule = pd.read_csv(results_folder + 'schedule.csv', index_col=0)
schedule = schedule[~(schedule.UserType == 'School')]
schedule.User = schedule.User.astype(str)

#%% 
hh = pd.DataFrame(schedule.groupby('User').TripDistance.sum())
hh.columns = ['tripd']
hh['hh'] = [int(float(i)) for i in hh.index]
idxhh = hh.groupby('hh').tripd.idxmax()

vide = [i for i in range(1,301) if not (i in idxhh.index)]

folds = ['NoToU_highplug',  'Asyn_ToU_highplug_ovn_v2']

for f in folds:
    util.create_folder(output_folder + f)
    
for f in folds:
    print(f)
    for ff in os.listdir(results_folder + f):
        if 'ev_profiles' in ff:
            print('\t' + ff)
            data = pd.read_csv(results_folder + f + r'\\' + ff, index_col=0)
            data.columns = [str(float(c)) for c in data.columns]
            data = data[idxhh]
            data.columns = idxhh.index
            for v in vide:
                data[v] = 0
            data = data.T.sort_index().T
            data.to_csv(output_folder + f + r'\\' + ff, index=False)
            