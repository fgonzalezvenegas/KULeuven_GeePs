# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:30:24 2020

@author: U546416
"""

# analyzing load profiles KUL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import EVmodel
import util_occupancy
import os
import util

# CHANGE THIS TO SELECT THE OCCUPANCY PROFILES FOLDER/FILE
folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimFiles\\'
file = 'OccupancyTotal.csv'
folder_loads = folder + 'Inputs for grid simulations\\'
folder_hp_load = folder + 'Re-ordered profiles\HPprofiles\\'
folder_pv = folder + 'Re-ordered profiles\PVprofiles\\'

# output folder for EV results
output_folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimResults\ToU_systematic\\'
base_case =  r'c:\user\U546416\Documents\PhD\Data\KULeuven\SimResults\NoToU_highplug\\'
# CASE
# just a keyword

# Reading files
print('Reading file')
print('\tOccupancy')
occupancy, type_member, household = util_occupancy.read_occupancy(folder + file) 
#there are some errors with type members that have an space
type_member = type_member.apply(lambda x: x.replace(' ', ''))

print('\tLoads')
# Reading base loads:
base = pd.read_csv(folder_loads +'P2.txt', 
                   sep='\t', skiprows=3, engine='python', 
                   header=None, index_col=0)
util_occupancy.index_to_dh(base)

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
    
# PV files

print('\tEV')
evfiles = os.listdir(output_folder)
evloads = {}
for f in evfiles:
    n = f.replace('.csv','').replace('_ev_profiles', '')
    if 'data' in n:
        continue
    evloads[n] = pd.read_csv(output_folder + f,
                            sep=',', engine='python', 
                            index_col=0)
    evloads[n].index = base.index

evloads['no_tou'] = pd.read_csv(base_case + 'none_ev_profiles.csv',
                            sep=',', engine='python', 
                            index_col=0)
evloads['no_tou'].index = base.index
#    evloads[n].columns = base.columns    


#%% Plot Heat Pump conso graphs (avg daily profile, daily conso)
plt.figure()
plt.title('Average heat pump profile [kW]')
for n, l in hploads.items():
    plt.plot(l.groupby('hour').mean().mean(axis=1)/1000, label=n)
plt.xlabel('Hour')
plt.legend()
plt.figure()
plt.title('Average daily consumption [kWh]')
for n, l in hploads.items():
    plt.plot(l.groupby('day').mean().mean(axis=1)/1000, label=n)
plt.xlabel('Day')
plt.legend()

#plot base conso
plt.figure()
plt.title('Average base load profile [kW]')
plt.plot(base.groupby('hour').mean().mean(axis=1)/1000)
plt.xlabel('Hour')
plt.legend()
plt.figure()
plt.title('Average base load daily consumption [kWh]')
plt.plot(base.groupby('day').mean().mean(axis=1)/1000)
plt.xlabel('Day')
plt.legend()

#plot total load
plt.figure()
plt.title('Average total profile [kW]')
for n, l in hploads.items():
    plt.plot((l+base).groupby('hour').mean().mean(axis=1)/1000, label=n)
plt.xlabel('Hour')
plt.legend()
plt.figure()
plt.title('Average daily consumption [kWh]')
for n, l in hploads.items():
    plt.plot((base+l).groupby('day').mean().mean(axis=1)/1000, label=n)
plt.xlabel('Day')
plt.legend()

#%% Max load per household and annual load histogram
plt.figure()
plt.hist((base).max(), bins=np.arange(1,16)*1000, label='No HP', alpha=0.2)
for n, l in hploads.items():
    plt.hist((base+l).max(), bins=np.arange(1,16)*1000, label=n, alpha=0.2)
plt.title('Distribution of max load')
plt.legend()
plt.xlabel('Load [W]')
plt.ylabel('# of houses')

           
plt.figure()
maxcap = 9000
plt.hist(((base )>maxcap).sum()/6, bins=np.arange(0,60), label='No HP', alpha=0.2)
for n, l in hploads.items():
    plt.hist(((base + l)>maxcap).sum()/6, bins=np.arange(0,60), label=n, alpha=0.2)
plt.title('Hours over 9kW')
plt.legend()
plt.xlabel('# of hours')
plt.ylabel('# of houses')

# Annual load histogram           
plt.figure()
k = 10000
bins=np.arange(0,k+1,500)
x = (bins[:-1]+bins[1:])/2
hb, _ = np.histogram((base).sum(axis=0)/(6*1000), bins=bins)
hhp, _ = np.histogram(np.concatenate([(l.sum(axis=0)/(6*1000)).values for l in hploads.values()]), bins=bins)
hev, _ = np.histogram((evloads['none_base'].sum(axis=0)/6).values, bins=bins)
plt.plot(x, hb/hb.sum(), label='Base load')
plt.plot(x, hhp/hhp.sum(), label='Heat pumps')
plt.plot(x, hev/hev.sum(), label='EV')

plt.title('Annual load [kWh]')
plt.legend()
plt.xlabel('Annual load [kWh]')
plt.ylabel('% of houses')
plt.xlim(0,k)
plt.ylim(0,0.25)
plt.grid()

#%% Plot annual load per LCT, violin plot
f,ax = plt.subplots()
data = [(base.sum(axis=0)/(6*1000)).values, 
        np.concatenate([(l.sum(axis=0)/(6*1000)).values for l in hploads.values()]),
        (evloads['none'].sum(axis=0)/6).values,
        np.concatenate([-(l.sum(axis=0)/(6*1000)).values for l in pvloads.values()])]
parts = plt.violinplot(data, showmedians=True)
colors = ['royalblue', 'darkorange', 'green', 'yellow']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.8)
parts['cbars'].set_color('k')
parts['cmaxes'].set_color('k')
parts['cmins'].set_color('k')
parts['cmedians'].set_color('r')
labels=  ['Base load', 'HP', 'EV', 'PV']
plt.xticks([1,2,3,4], labels)
#plt.title('Annual load [kWh]')
plt.legend()
plt.ylabel('Annual load [kWh]')
plt.grid(axis='y', linestyle='--')
f.tight_layout()

##%%
## read results
#evfiles = os.listdir(output_folder + case)
#
#print('reading EV profiles from case {}'.format(case))
#evprofs= {}
#for f in evfiles:
#    n = f.replace('_ev_profiles.csv','')
#    print('\t{}'.format(n))
#    evprofs[n] = pd.read_csv(output_folder + case + r'\\'+ f,
#                            sep=',', skiprows=0, engine='python', 
#                            header=0, index_col=0)
#    evprofs[n].index = base.index
##    evprofs[n].columns = base.columns
 
#%% Plot one week for a single house, CIRED paper
# type 1, worst home
# starting day
d=13

f,axs = plt.subplots(2)

# selecting house
lct = 'HP'
typ = 'type1'
l = len(evloads[lct + typ].iloc[:,0].values)
# selecting house with highest HP+baseload conso (in kWh/year)
i = (base + hploads[typ]).sum(axis=0).idxmax()
i = (base + hploads[typ]).sum(axis=0).sort_values().index[-2]
#i = np.random.choice(base.columns)
x = np.arange(0,l/24/6,1/24/6)
pfhp = 0.98
pfev = 1
pfpv = 1
pfbl = .98
start = 1
xminor = np.arange(0,l/24/6,1/4)
xlabelminor = np.tile(np.arange(0,24,6),365)


# Plot dumb case
ax = axs[0]
plt.sca(ax)
ax.set_xticks(np.arange(0,365))
ax.set_xticklabels(np.tile(util.dsnms, 53)[start:])
for t in ax.get_xticklabels(minor=False):
    xy = t.get_position()
    t.set_horizontalalignment('left')
    t.set_y(0)
ax.set_xticks(xminor, minor=True)
ax.set_xticklabels(xlabelminor, minor=True)
for t in ax.get_xticklabels(minor=True):
    t.set_y(-0.05)

plt.stackplot(x, [pvloads[typ].loc[:,i].values/1000/pfpv, 
                  base.loc[:,i].values/1000/pfbl, 
                  hploads[typ].loc[:,i].values/1000/pfhp, 
                  evloads['no_tou'].loc[:,str(float(i))].values/pfev],#*7.4/7], 
            labels=['_PV', '_Base load', '_Heat Pump', '_EV load'],
            colors=['yellow', 'royalblue', 'darkorange', 'g'])
ax.axhline(9.2, linestyle='--', color='r', label='_Contracted power')

plt.xlim([d,d+7-0.01])
plt.yticks(np.arange(-6,18,3))
plt.ylim([-3,16])
plt.grid()
#plt.xlabel('Day/Hour')
plt.ylabel('Power [kVA]')

# Plot smart case
ax = axs[1]
plt.sca(ax)
# plot labels and ticks
ax.set_xticks(np.arange(0,365))
ax.set_xticklabels(np.tile(util.dsnms, 53)[start:])
for t in ax.get_xticklabels(minor=False):
    xy = t.get_position()
    t.set_horizontalalignment('left')
    t.set_y(0)
ax.set_xticks(xminor, minor=True)
ax.set_xticklabels(xlabelminor, minor=True)
for t in ax.get_xticklabels(minor=True):
    t.set_y(-0.05)
ax.tick_params(axis='x', which='major', direction='out')

#Plotting
plt.stackplot(x, [pvloads[typ].loc[:,i].values/1000/pfpv, 
                  base.loc[:,i].values/1000/pfbl, 
                  hploads[typ].loc[:,i].values/1000/pfhp, 
                  evloads[lct+typ].loc[:,str(float(i))].values/pfev],#*7.4/7], 
            labels=['PV', 'Base load', 'HP', 'EV'],
            colors=['yellow', 'royalblue', 'darkorange', 'g'])
ax.axhline(9.2, linestyle='--', color='r', label='Contracted power')
# Setting view for interesting week
plt.xlim([d,d+7-0.01])
plt.yticks(np.arange(-6,18,3))
plt.ylim([-3,16])
plt.grid()
#plt.xlabel('Day/Hour')
plt.ylabel('Power [kVA]')
#plt.legend(loc=1)



#plt.legend(loc=1)
#f.set_size_inches([ 6.9,  8.2])
f.set_size_inches(6,7)
#f.set_size_inches([ 5.36,  6.31])
f.tight_layout()
   
f.legend(ncol=3, loc=8)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
#    dy = 0.015
    dy = 0.03
    ax.set_position([pos.x0, pos.y0+dy*(i+1), pos.width, pos.height-dy])

#plt.plot(, (base+hploads[t3]).iloc[:,0].values/1000)
#plt.plot(evprofs[t3].index, (base+hploads[t3]).iloc[:,0].values/1000 + evprofs[t3].iloc[:,0].values)

#%% Plot loads per week

bl = base.mean(axis=1)[:-(1+24*6)]/1000
ev = evloads['no_tou'].mean(axis=1)[:-(1+24*6)]
hp = np.stack([h.mean(axis=1) for h in hploads.values()]).mean(axis=0)[:-(1+24*6)]/1000
pv = np.stack([h.mean(axis=1) for h in pvloads.values()]).mean(axis=0)[:-(1+24*6)]/1000

blwe = bl.reshape(52,7*24*6).sum(axis=1)/6
evwe = ev.reshape(52,7*24*6).sum(axis=1)/6
hpwe = hp.reshape(52,7*24*6).sum(axis=1)/6
pvwe = pv.reshape(52,7*24*6).sum(axis=1)/6
xw = np.arange(0,52)

ls = ['-','--',':','-.']

plt.figure()
plt.plot(xw, blwe, label='Base load', color='royalblue',linestyle=ls[2], linewidth=2)
plt.plot(xw, evwe, label='EV', color='g',  linestyle=ls[0])
plt.plot(xw, hpwe, label='HP', color='darkorange', linestyle=ls[1])
plt.plot(xw, -pvwe, label='PV', color='gold', linestyle=ls[3])
plt.xlim(0,51)
plt.grid(linestyle='--')
plt.axhline(0, color='k', alpha=0.5, linestyle='--')
plt.legend()
plt.ylabel('Weekly load [kWh]')
xt  = np.array([0,31,28,31,30,31,30,31,31,30,31,30]).cumsum()/364*52
#plt.xticks(np.linspace(1,52,13), util.monthnames)
plt.xticks(xt, util.monthnames)
ax= plt.gca()
for t in ax.get_xticklabels(minor=False):
    t.set_horizontalalignment('left')
plt.tight_layout()
    
#%% Plot equivalent load at trafo
#
nhouses = 78
while True:
    hs = np.random.choice(base.columns, nhouses,replace=False)
    if not 33 in hs:
        break
lct = 'HP'
typ = 'type1'
l = len(evloads[lct + typ].iloc[:,0].values)
losses = 10 #percentage

x = np.arange(0,l/24/6,1/24/6)
start = 1
xminor = np.arange(0,l/24/6,1/4)
xlabelminor = np.tile(np.arange(0,24,6),365)

pfhp = 0.98
pfev = 1
pfpv = 1
pfbl = .98
    
f,axs = plt.subplots(2)

ax=axs[0]
ax.set_xticks(np.arange(0,365))
ax.set_xticklabels(np.tile(util.dsnms, 53)[start:])
for t in ax.get_xticklabels(minor=False):
    xy = t.get_position()
    t.set_horizontalalignment('left')
    t.set_y(0)
ax.set_xticks(xminor, minor=True)
ax.set_xticklabels(xlabelminor, minor=True)
for t in ax.get_xticklabels(minor=True):
    t.set_y(-0.05)

ax.stackplot(x, np.array([pvloads[typ][hs].sum(axis=1)/1000/pfpv, 
                          base[hs].sum(axis=1)/1000/pfbl, 
                          hploads[typ][hs].sum(axis=1)/1000/pfhp, 
                          evloads['no_tou'][[str(float(s)) for s in hs]].sum(axis=1)
                          ])*(1 + losses/100),
#              labels=['PV', 'Base load', 'HP', 'EV load'],
            colors=['yellow','royalblue',  'darkorange', 'g'])
ax.axhline(160, color='r', linestyle='--')
d=6
ax.set_xlim([d,d+7])
#plt.yticks(np.arange(-6,18,3))
ax.set_ylim([-2*nhouses,5*nhouses])
ax.grid()
#plt.xlabel('Day/Hour')
ax.set_ylabel('Power [kVA]')



ax=axs[1]
ax.set_xticks(np.arange(0,365))
ax.set_xticklabels(np.tile(util.dsnms, 53)[start:])
for t in ax.get_xticklabels(minor=False):
    xy = t.get_position()
    t.set_horizontalalignment('left')
    t.set_y(0)
ax.set_xticks(xminor, minor=True)
ax.set_xticklabels(xlabelminor, minor=True)
for t in ax.get_xticklabels(minor=True):
    t.set_y(-0.05)


ax.stackplot(x, np.array([pvloads[typ][hs].sum(axis=1)/1000/pfpv, 
                          base[hs].sum(axis=1)/1000/pfbl, 
                          hploads[typ][hs].sum(axis=1)/1000/pfhp, 
                          evloads[lct+typ][[str(float(s)) for s in hs]].sum(axis=1)
                          ])*(1 + losses/100),
              labels=['PV', 'Base load', 'HP', 'EV load'],
            colors=['yellow','royalblue',  'darkorange', 'g'])
ax.axhline(160, color='r', linestyle='--', label='Transformer rated power')
d=6
ax.set_xlim([d,d+7])
#plt.yticks(np.arange(-6,18,3))
ax.set_ylim([-2*nhouses,5*nhouses])
ax.grid()
#plt.xlabel('Day/Hour')
ax.set_ylabel('Power [kVA]')
#ax.legend(loc=1)
#f.set_size_inches([ 7.49,  4.76])
f.set_size_inches(6,7)
#f.set_size_inches([ 5.36,  6.31])
f.tight_layout()
   
f.legend(ncol=3, loc=8)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
#    dy = 0.015
    dy = 0.03
    ax.set_position([pos.x0, pos.y0+dy*(i+1), pos.width, pos.height-dy])


#%% Plot equivalent load at trafo, without HP
nhouses = 78
while True:
    hs = np.random.choice(base.columns, nhouses,replace=False)
    if not 33 in hs:
        break
lct = 'HP'
typ = 'type1'
losses = 10 #percentage    

#id the max ev load case
day,_= evloads[lct+typ][[str(float(s)) for s in hs]].sum(axis=1).idxmax()
d = (day//7)*7

#

f,axs = plt.subplots(2)
# Plots the uncontrolled case
# setting axis ticks
ax=axs[0]
ax.set_xticks(np.arange(0,365))
ax.set_xticklabels(np.tile(util.dsnms, 53)[start:])
for t in ax.get_xticklabels(minor=False):
    xy = t.get_position()
    t.set_horizontalalignment('left')
    t.set_y(0)
ax.set_xticks(xminor, minor=True)
ax.set_xticklabels(xlabelminor, minor=True)
for t in ax.get_xticklabels(minor=True):
    t.set_y(-0.05)

#plotting
ax.stackplot(x, np.array([pvloads[typ][hs].sum(axis=1)/1000/pfpv, 
                          base[hs].sum(axis=1)/1000/pfbl, 
                          evloads['no_tou'][[str(float(s)) for s in hs]].sum(axis=1)
                          ])*(1 + losses/100),
#              labels=['PV', 'Base load', 'HP', 'EV load'],
            colors=['yellow','royalblue',  'g'])
ax.axhline(160, color='r', linestyle='--')
ax.axhline(-160, color='r', linestyle='--', label='_Transformer rated power')
ax.set_xlim([d,d+7])
#plt.yticks(np.arange(-6,18,3))
ax.set_ylim([-4*nhouses,4*nhouses])
ax.grid()
#plt.xlabel('Day/Hour')
ax.set_ylabel('Power [kVA]')


# plotting smart charge case
# setting ticks
ax=axs[1]
ax.set_xticks(np.arange(0,365))
ax.set_xticklabels(np.tile(util.dsnms, 53)[start:])
for t in ax.get_xticklabels(minor=False):
    xy = t.get_position()
    t.set_horizontalalignment('left')
    t.set_y(0)
ax.set_xticks(xminor, minor=True)
ax.set_xticklabels(xlabelminor, minor=True)
for t in ax.get_xticklabels(minor=True):
    t.set_y(-0.05)


ax.stackplot(x, np.array([pvloads[typ][hs].sum(axis=1)/1000/pfpv, 
                          base[hs].sum(axis=1)/1000/pfbl, 
                          evloads[lct+typ][[str(float(s)) for s in hs]].sum(axis=1)
                          ])*(1 + losses/100),
              labels=['PV', 'Base load', 'EV load'],
            colors=['yellow','royalblue', 'g'])
ax.axhline(160, color='r', linestyle='--', label='Transformer rated power')
ax.axhline(-160, color='r', linestyle='--', label='_Transformer rated power')
ax.set_xlim([d,d+7])
#plt.yticks(np.arange(-6,18,3))
ax.set_ylim([-4*nhouses,4*nhouses])
ax.grid()
#plt.xlabel('Day/Hour')
ax.set_ylabel('Power [kVA]')
#ax.legend(loc=1)
#f.set_size_inches([ 7.49,  4.76])
f.set_size_inches(6,7)
#f.set_size_inches([ 5.36,  6.31])
f.tight_layout()
   
f.legend(ncol=3, loc=8)
# resizing axs to leave space for legend
for i, ax in enumerate(axs):
    pos = ax.get_position()
#    dy = 0.015
    dy = 0.03
    ax.set_position([pos.x0, pos.y0+dy*(i+1), pos.width, pos.height-dy])