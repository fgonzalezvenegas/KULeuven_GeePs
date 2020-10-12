# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:14:15 2020
Script to run EV model based on occupancy profiles from KU Leuven
@author: U546416
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import EVmodel
import util_occupancy
import os

###########################################################################################

#################                INPUT DATA                                 ###############

###########################################################################################

# CHANGE THIS TO SELECT THE OCCUPANCY PROFILES FOLDER/FILE
folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimFiles\\'
file = 'OccupancyTotal.csv'
folder_base_load = folder + 'Inputs for grid simulations\\'
folder_hp_load = folder + 'Inputs for grid simulations\HPprofiles\\'

# output folder for results
output_folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimResults\\'

# CASE
# just a keyword
case = 'NoToU'

###################################  EV params   ##########################################

batt = 50
pcharge = 7
alpha_plug = 0.15
charging_type='if_needed'

evparams = dict(batt_size=batt,
                charging_power=pcharge,
                n_if_needed=alpha_plug,
                charging_type=charging_type)

# Contracted capacity kW for each household
contracted_cap = 9

# Cases

all_cases = True



##########################################################################################



#%% Reading files
print('Reading files')
print('\tOccupancy')
occupancy, type_member, household = util_occupancy.read_occupancy(folder + file) 
#there are some errors with type members that have an space
type_member = type_member.apply(lambda x: x.replace(' ', ''))

print('\tLoads')
# Reading base loads:
base = pd.read_csv(folder_base_load +'P2.txt', 
                   sep='\t', skiprows=3, engine='python', 
                   header=None, index_col=0)
pv = pd.read_csv(folder_base_load + 'pv_profiles_RSmaxTWmin.txt',
                  sep=',', skiprows=6, engine='python', 
                  header=None, index_col=0)
util_occupancy.index_to_dh(base)
pv.index = base.index

# hp files
hpfiles = os.listdir(folder_hp_load)

hploads = {}
for f in hpfiles:
    n = f.replace('.txt','')
    hploads[n] = pd.read_csv(folder_hp_load + f,
                            sep=',', skiprows=2, engine='python', 
                            header=None, index_col=[0,1])
    hploads[n].index = pv.index
    hploads[n].columns = base.columns

# creating load profiles for EV users in kW:
# total0 is base load
# total1,2,3 is base load + HP1,2,3

totalloads = {'none' : 0}
totalloads['base'] = base[household] / 1000
for i, h in hploads.items():
    totalloads[i] = (base[household] + h[household])/ 1000
    totalloads[i].columns = occupancy.columns
# add pv?


# Usefull data
allusers =  ['FTE', 'PTE', 'Retired', 'Unemployed', 'School']
activeusers = ['FTE', 'PTE', 'Retired', 'Unemployed']
evusers = type_member[type_member.isin(activeusers)].index 
evocc = occupancy[evusers]

# First day of occupancy profiles (needed for trip motifs)
day_ini = 1 # 0 is monday. KULeuven data starts on Tuesday (1)

# Compute EV schedule from occupancy model
print('Computing schedule')
schedule = util_occupancy.do_schedule(occupancy, type_member, weekday_ini=day_ini)

# Compute travel distances for each trip
print('Computing distances')
util_occupancy.set_distances_to_schedule(schedule, verbose=True)

# Plot avg daily distance per EV user
util_occupancy.plot_user_avg_dist_hist(schedule)
plt.savefig(output_folder + case + '_AverageUserDist.png')
# save schedule
schedule.to_csv(output_folder + case + '_schedule.csv')

#%% Simulate grid


# grid params:
# number of days in occupancy profiles
ndays_occ = schedule.ArrDay.max() + 1
days_before = 7
days_after = 1
step = 10 # minutes

# Copy travel dists to charging schedule:
util_occupancy.set_dist_to_charging_sessions(schedule)
## Repeat schedule ntimes 
#nrepeat = ndays // ndays_occ
#fullsch = util_occupancy.repeat_schedule(schedule, nrepeat)
# Add before and after days
fullsch = util_occupancy.add_days(schedule, days_before, days_after)

# Extract only charging sessions
chsessions = util_occupancy.get_charging_sessions(fullsch)
print('Null charging sessions {}'.format(chsessions.TripDistance.isnull().sum()))
#%%

# create grid
grid = EVmodel.Grid(ndays=ndays_occ + days_before + days_after, 
                    step=step, init_day=day_ini, name=case)

# add evs
for u in activeusers:
    nevs = chsessions[chsessions.UserType == u].User.nunique() # number of EVs of that type
    grid.add_evs(u, nevs, 'dumb', 
                 charge_schedule=chsessions[chsessions.UserType == u],
                 **evparams)


for i in totalloads:
    print('Doing case ' + i)
    # TODO: adapt to do all cases
        # cases: No load, base load, base + HP1, Base+HP2, Base+HP3
    if totalloads[i] != 0:
        # adding limit load
        grid.add_evparam_from_dataframe('vcc', totalloads[i])
    
    fn = case + '_' + i + '_'
    
    grid.do_days()
    
    
    grid.plot_ev_load(day_ini=7, days=3*7)
    plt.gca().figure.set_size_inches([18.71, 4.76])
    plt.savefig(output_folder + fn + '_evload.png')
    grid.hist_ncharging()
    plt.savefig(output_folder + fn + '_nplugs_hist.png')
    ev_data = grid.get_ev_data()
    results = ['avg_plugin', 'charge', 'extra_charge']
    data = ''
    for i, sets in enumerate(ev_data['EV_sets']):
        data = data + '{:10},\t'.format(sets)
    print('{:17}'.format('KPI') + '\t: ' + data)
    for kpi in results:
        data = ''
        for i in ev_data[kpi]:
            data = data + '{:10.2f},\t'.format(i)
        print('{:17}'.format(kpi) + '\t: ' + data)
        
    # Save results
    # folder & file
    
    filename = 'ev_profiles.csv'
    
    # extracting charging
    print('Extracting charging profiles')
    evs = grid.get_evs()
    ev_load = pd.DataFrame([ev.charging for ev in evs], index=[ev.name for ev in evs]).T
    
    # Dropping first week and last day
    ev_load = ev_load.iloc[days_before*24*6:-24*6*days_after].reset_index(drop=True)
    
    ## Changing names
    #cols = ev_load.columns
    #for t in allusers:
    #    cols = [c.replace(t,'') for c in cols]
    #ev_load.columns = cols
    
    # saving
    print('Saving at '+ output_folder + fn + filename)
    ev_load.to_csv(output_folder + case + fn + filename)
    # reset 
    grid.reset()