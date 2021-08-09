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

# output folder for results
output_folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimResults\\'

# CASE
# just a keyword
case = 'ToU_systematic'
util.create_folder(output_folder + case)
util.create_folder(output_folder + case + '_Images')

###################################  EV params   ##########################################

batt = 50
pcharge = 7
alpha_plug = 0.5
charging_type='all_days'
tou_ini = 23
tou_end = 8
driving_eff = 0.18

evparams = dict(batt_size=batt,
                charging_power=pcharge,
                n_if_needed=alpha_plug,
                charging_type=charging_type,
                tou_ini=tou_ini,
                tou_end=tou_end,
                tou_we = True,
                tou_ini_we=tou_ini,
                tou_end_we=tou_end,
                driving_eff=driving_eff)

asynchronous_tou = True
asyn_tou_ini = 0
asyn_tou_step = 1
asyn_tou_end = 4
asyn_tou_len = 8

# Contracted capacity [kW] for each household
contracted_cap = 9

# Cases

base_case = True
hp = True
pv = True

do_schedule = False
##########################################################################################

tou = (not (tou_ini == tou_end)) or asynchronous_tou

#%% Reading files
print('Reading files')
print('\tOccupancy')
if do_schedule:
    occupancy, type_member, household = util_occupancy.read_occupancy(folder + file) 
    #there are some errors with type members that have an space
    type_member = type_member.apply(lambda x: x.replace(' ', ''))
else:
    # Reading pre-computed schedules
    schedule = pd.read_csv(output_folder + 'schedule.csv', index_col=0)
    schedule = schedule[~(schedule.UserType == 'School')]
    schedule.User = schedule.User.astype(str)
    ovn_schedule = pd.read_csv(output_folder + 'ovn_schedule.csv', index_col=0)
    ovn_schedule = ovn_schedule[~(ovn_schedule.UserType == 'School')]
    ovn_schedule.User = ovn_schedule.User.astype(str)
    type_member = schedule.groupby(['User','UserType']).size().reset_index().set_index('User', drop=True).drop(0,axis=1)    
    household = pd.Series([int(float(i)) for i in schedule.User.unique()], index=schedule.User.unique())
    
print('\tLoads')
# Reading base loads:
base = pd.read_csv(folder_base_load +'P2.txt', 
                   sep='\t', skiprows=3, engine='python', 
                   header=None, index_col=0)
util_occupancy.index_to_dh(base)


# hp files
if hp:
    print('\tHeat Pumps')
    hpfiles = os.listdir(folder_hp_load)
    
    hploads = {}
    for f in hpfiles:
        n = f.replace('.txt','')
        hploads[n] = pd.read_csv(folder_hp_load + f,
                                sep=',', skiprows=2, engine='python', 
                                header=None, index_col=0)
        hploads[n].index = base.index
        hploads[n].columns = base.columns
    
    
# PV files
if pv:
    print('\tPV')
    pvfiles = os.listdir(folder_pv)
    
    pvloads = {}
    for f in pvfiles:
        n = f.replace('.txt','').replace('PV', '')
        pvloads[n] = pd.read_csv(folder_pv + f,
                                sep=',', skiprows=3, engine='python', 
                                header=None, index_col=0)
        pvloads[n].index = base.index
        pvloads[n].columns = base.columns


# creating load profiles for EV users in kW:
# total0 is base load
# total1,2,3 is base load + HP1,2,3

totalloads = {'none' : 0}
if base_case:
    totalloads['base'] = base[household] / 1000
    totalloads['base'].columns = household.index
if hp:
    for i, h in hploads.items():
        totalloads['HP' + i] = (base[household] + h[household])/ 1000
        totalloads['HP' + i].columns = household.index
if pv:
    for i, p in pvloads.items():
        totalloads['PV' + i] = (base[household] + p[household])/ 1000
        totalloads['PV' + i].columns = household.index
if pv and hp:
    for i, p in pvloads.items():        
        totalloads['HP_PV' + i] = (base[household] + hploads[i][household] + p[household])/ 1000
        totalloads['HP_PV' + i].columns = household.index
        
# add pv?


# Usefull data
allusers =  ['FTE', 'PTE', 'Retired', 'Unemployed', 'School']
activeusers = ['FTE', 'PTE', 'Retired', 'Unemployed']
evusers = type_member[type_member.isin(activeusers)].index 
#evocc = occupancy[evusers]

# First day of occupancy profiles (needed for trip motifs)
day_ini = 1 # 0 is monday. KULeuven data starts on Tuesday (1)

if do_schedule:
    # Compute EV schedule from occupancy model
    print('Computing schedule')
    schedule = util_occupancy.do_schedule(occupancy, type_member, weekday_ini=day_ini)
    
    # Compute travel distances for each trip
    print('Computing distances')
    util_occupancy.set_distances_to_schedule(schedule, verbose=True)
    
    # Plot avg daily distance per EV user
    util_occupancy.plot_user_avg_dist_hist(schedule)
    plt.savefig(output_folder + case + r'_Images\AverageUserDist.png')
    
    # Computing schedule with only ovn charging
    print('Computing overnight schedule')
    ovn_schedule =  util_occupancy.modify_schedule(schedule, blocked_times=[8,21])
    
    # save schedule
    schedule.to_csv(output_folder + case + r'\schedule.csv')
    household.to_csv(output_folder + case + r'\household.csv')
    ovn_schedule.to_csv(output_folder + case + r'\ovn_schedule.csv')
    
print('Adapting schedule to grid simulation')
# number of days in occupancy profiles
ndays_occ = schedule.ArrDay.max() + 1
days_before = 7
days_after = 1
step = 10 # minutes

# Copy travel dists to charging schedule:
util_occupancy.set_dist_to_charging_sessions(schedule)
util_occupancy.set_dist_to_charging_sessions(ovn_schedule)
## Repeat schedule ntimes 
#nrepeat = ndays // ndays_occ
#fullsch = util_occupancy.repeat_schedule(schedule, nrepeat)
# Add before and after days
fullsch = util_occupancy.add_days(schedule, days_before, days_after)
fullovn = util_occupancy.add_days(ovn_schedule, days_before, days_after)

# Extract only charging sessions
chsessions = util_occupancy.get_charging_sessions(fullsch)
chsessions_ovn = util_occupancy.get_charging_sessions(fullovn)

# Adding one week before and one day after
for i, t in totalloads.items():
    if i == 'none':
        continue
    totalloads[i] = pd.concat([t.iloc[0:7*24*6,:] * 0, t, t.iloc[-1*24*6:,:]], ignore_index=True)

        

print('Null charging sessions {}'.format(chsessions.TripDistance.isnull().sum()))
#%%
t0 = time.time()
# create grid
grid = EVmodel.Grid(ndays=ndays_occ + days_before + days_after, 
                    step=step, init_day=day_ini, name=case)

# add evs
for u in activeusers:
    nevs = chsessions[chsessions.UserType == u].User.nunique() # number of EVs of that type
    chs = chsessions[chsessions.UserType == u] #if not tou else chsessions_ovn[chsessions_ovn.UserType == u]
    
    grid.add_evs(u, nevs, 'dumb', 
                 charge_schedule=chs,
                 **evparams)

if asynchronous_tou:
    print('Creating parameters for asynchronous ToU')
    tou_ini={}
    tou_end={}
    a = 0
    for u in evusers:
        tou_ini[u] = (asyn_tou_ini + a)%24
        tou_end[u] = (asyn_tou_ini + a + asyn_tou_len)%24
        a += asyn_tou_step
        a = a%((asyn_tou_end-asyn_tou_ini)%24)
    grid.add_evparam_from_dict('tou_ini', tou_ini)
    grid.add_evparam_from_dict('tou_end', tou_end)
    grid.add_evparam_from_dict('tou_ini_we', tou_ini)
    grid.add_evparam_from_dict('tou_end_we', tou_end)

for tl in totalloads:
    print('Doing case ' + tl)
#    if tl in ['none']:
#        continue
    # TODO: adapt to do all cases
        # cases: No load, base load, base + HP1, Base+HP2, Base+HP3
    if not (totalloads[tl] is 0):
        # adding limit load
        print('Adding capacity limit')
        grid.add_evparam_from_dataframe('vcc', (contracted_cap - totalloads[tl]).clip(lower=0))
    
    fn = case + r'\\' + tl + '_'
    fim = case + r'_Images\\' + tl
    
    grid.do_days()
    
    
    grid.plot_ev_load(day_ini=7, days=3*7)
    plt.gca().figure.set_size_inches([18.71, 4.76])
    plt.savefig(output_folder + fim + '_evload.png')
#    grid.hist_ncharging()
#    plt.savefig(output_folder + fim + '_nplugs_hist.png')
    ev_data = grid.get_ev_data()
    results = ['Avg_plug_in_ratio', 'EV_charge_MWh', 'Extra_charge']
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
    
    # Dropping first week and last day, with 1 extra step
    ev_load = ev_load.iloc[days_before*24*6:(-24*6*days_after+1)].reset_index(drop=True)
    
    ## Changing names
    #cols = ev_load.columns
    #for t in allusers:
    #    cols = [c.replace(t,'') for c in cols]
    #ev_load.columns = cols
    
    # saving
    print('Saving at '+ output_folder + fn + filename + '\n')
    ev_load.to_csv(output_folder + fn + filename)
    ev_data =  pd.DataFrame(ev_data).set_index("EV_sets", drop=True).to_csv(output_folder + fn + 'ev_data.csv')
    # reset 
    grid.reset()
    
print('Finished all sims. Total elapsed time {}h {:02d}:{:04.1f}'.format(*util.sec_to_time((time.time()-t0))))
