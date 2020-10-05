# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:28:05 2020

@author: U546416
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import util
import util_occupancy


# CHANGE THIS TO SELECT THE OCCUPANCY PROFILES FOLDERE/FILE
folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\OccupancyProfiles\\'
#file = 'Occupancy.csv'
file = 'Occupancy_1000_good.csv'

# You get pandas dataframe of occupancy, and series with Member type (FTE, etc) and household number
occupancy, type_member, household = util_occupancy.read_occupancy(folder + file) 

#
allusers =  ['FTE', 'PTE', 'Retired', 'Unemployed', 'School']
activeusers = ['FTE', 'PTE', 'Retired', 'Unemployed']
evusers = type_member[type_member.isin(activeusers)].index 
evocc = occupancy[evusers]


##%% Plot occupancy of one house
##
##plt.plot(evocc.iloc[:,0])
##plt.xticks(np.arange(0,7*60*24/10,6*4), [int(i) for i in (np.arange(0,7*60*24/10,6*4)*10/60)%24])
##
#days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#plt.subplots()
#
#plt.plot(((occupancy['2']<3)*1).values,'og', markersize=2 ,label='Home', alpha=0.7)
#plt.plot(((occupancy['2']==3)*1).values,'ob', markersize=4, label='Away')
#for i in range(1,4):
#    plt.plot(((occupancy['2.{}'.format(i)] <3)*(i+1)).values,'og', alpha=0.7, markersize=2 )
#    plt.plot(((occupancy['2.{}'.format(i)]==3)*(i+1)).values,'ob', markersize=4)
#
#ticks = np.arange(0,7*60*24/10,6*6)
#tickslab = [str(int(i%24))+'\n{}'.format(days[i//24]) if (i%24)==12 else str(int(i%24))
#                 for i in np.arange(0,7*24+1,6)]
#plt.xticks(ticks, tickslab)
#plt.ylim(0.5,4.5)
#plt.xlim(-1,24*7*6+1)
#plt.grid(alpha=0.5)
#plt.legend()
#plt.title('Home occupancy')
#plt.yticks([1,2,3,4], ['FTE-1', 'FTE-2', 'School-1', 'School-2'])
#
#
###%%
##plt.subplots()
##plt.plot(occupancy['2'].values)
##plt.xticks(ticks, tickslab)

#%% Total out time
# out time per day (should separate for type of people and wd/we)?

# ID of weekend weekday
wd = [0, 5*24*6]
sat = [5*24*6, 6*24*6]
sun = [6*24*6, 7*24*6]

for j, d in enumerate([wd, sat, sun]):
    plt.subplots()
    for u in activeusers:
        users = type_member[type_member == u].index 
        outtime = (evocc[users].iloc[d[0]:d[1]] == 3).sum(axis=0) / 6 / 7
        h,b = np.histogram(outtime, bins=np.arange(0,24,1))
        h = h/h.sum()
        plt.plot((b[:-1]+b[1:])/2, h, label=u)
        plt.title('Time spent away from home, '+ ['weekday', 'saturday', 'sunday'][j] )
        plt.xlabel('Time spent away [h]')
        plt.ylabel('Distribution')
        plt.legend()
#%% Average profiles at home
dt = 24*6
wd = [0,5]
sat = [5,6]
sun = [6,7]     
x = np.arange(0,24,1/6)

for j, d in enumerate([wd, sat, sun]):
    plt.subplots()
#    for u in activeusers:
    for u in allusers:
        users = type_member[type_member == u].index
        prof = (occupancy[users].iloc[d[0]*dt:d[1]*dt] < 3).mean(axis=1)
        if d[1]-d[0]>1:
            prof = prof.values.reshape(d[1]-d[0],dt).mean(axis=0)
        plt.plot(x, prof, label=u)
    #plot average
    users = type_member[type_member.isin(activeusers)].index
    prof  = (occupancy[users].iloc[d[0]*dt:d[1]*dt] < 3).mean(axis=1)
    if d[1]-d[0]>1:
            prof = prof.values.reshape(d[1]-d[0],dt).mean(axis=0)
    plt.plot(x, prof, '--', linewidth=3, label='All active users')
    # format
    plt.title('Average profile at home, ' + ['weekday', 'saturday', 'sunday'][j] )
    plt.xlabel('Time of day [h]')
    plt.ylabel('Probability of being at home')
    plt.legend()
    plt.xlim(0,24)
    plt.ylim(0,1)


#%% Print general data and plot household histogram
print('Share of profiles from total, {} unique profiles'.format(len(type_member)))
print(type_member.value_counts()/len(type_member)*100)

plt.figure()
household.value_counts().hist(bins=np.arange(1,7))
#plt.title('Number of persons per household')
plt.xlabel('Number of persons per household')
plt.ylabel('Frequency')
#%%Creating data frame of charging and trip sessions, 
df = util_occupancy.do_schedule(occupancy, type_member)

#df.to_csv(folder + 'sessions_1000_good.csv')

#%% Add distance based on Van Roy
df = pd.read_csv(folder + 'sessions_1000_good.csv', index_col=0)

import util
util.self_reload(util_occupancy)
util_occupancy.set_distances_to_schedule(df, verbose=True)

#df.to_csv(folder + 'sessions_1000_good.csv')
#%% Plot arrival & departures to/from home
days = ['Weekday', 'Sat', 'Sun']
bins = np.arange(0,24.5,0.5)

for w in days:
    dff = df[(df.DoW == w) & (df.AtHome) & (df.DeltaTime > 2)]
    h, _, _ = np.histogram2d(dff.ArrTime, dff.DepTime, bins=bins)
    util_occupancy.plot_arr_dep_hist(h, binsh=bins, ftitle='Arrival and Departure distributions, ' + w)
    
##%% Plot arrival departures
#days = ['Weekday', 'Sat', 'Sun']
#bins = np.arange(0,24.5,0.5)
#
#for w in days:
#    dff = df[(df.DoW == w) & (df.AtHome)]
#    h, _, _ = np.histogram2d(dff.InHour, dff.OutHour, bins=bins)
#    util.plot_arr_dep_hist(h, binsh=bins, ftitle='Arrival and Departure distributions, ' + w)
    
#%% Analysis of 'trips'
days = ['Weekday', 'Sat', 'Sun']
bins = np.arange(0,24.5,0.5)

for w in days:
    dff = df[(df.DoW == w) & (~df.AtHome)]
    h, _, _ = np.histogram2d(dff.ArrTime, dff.DeltaTime, bins=bins)
    f, (ax, ax2) = plt.subplots(1,2)
    i = ax.imshow(h.T/h.sum().sum(), origin='lower', extent=(0,24,0,24))
    ax.set_title('Distribution of start of trip vs trip length')
    ax.set_xlabel('Start of trip')
    ax.set_ylabel('Trip length [h]')
    ax.set_xticks(np.arange(0,25,2))
    ax.set_yticks(np.arange(0,25,2))
    ax.set_xticklabels(np.arange(0,25,2))
    ax.set_yticklabels(np.arange(0,25,2))
    plt.colorbar(i, ax=ax)
    
    ax2.bar((bins[:-1]+bins[1:])/2, h.sum(axis=0)/h.sum().sum(), width=0.5, label='Trip duration')
    ax2.set_xlim(0,24)
    ax2.set_xticks(np.arange(0,25,2))
    ax2.set_xticklabels(np.arange(0,25,2))
    ax2.set_title('Trip duration distribution')
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Distribution')
    ax2.legend()
    ax2.grid()
    f.suptitle('Trip duration distribution, {}'.format(w))
    f.set_size_inches(11.92,4.43)
    
    
#%% Analysis of trip distances
    
f,ax = plt.subplots(1,2)
# plot 1, histograms of trip distances
dx=2
bins = np.arange(0,101,dx)
trips = ['Work', 'Shopping', 'Recreation', 'Other']
plt.sca(ax[0])
x = np.arange(dx/2,100,dx)
hc=0
for i in trips:
    h, _ = np.histogram(df[(~df.AtHome) & (df.TripMotif == i)].TripDistance, bins=bins)
    pkm = df[(~df.AtHome) & (df.TripMotif == i)].TripDistance.sum() / df.TripDistance.sum()*100
    plt.bar(x, h, width=dx, bottom=hc, label=i+' ({:.1f}% of kms)'.format(pkm))
    hc += h
plt.legend()
plt.xlabel('Trip distance [km]')
plt.ylabel('Frequency')
plt.title('Trip distance distribution')

plt.sca(ax[1])

userdist = df[~df.AtHome].groupby('User').TripDistance.sum() / 7
avgd = userdist.mean()
h, _ = np.histogram(userdist, bins=bins)
plt.bar(x, h, width=5)
plt.axvline(avgd, color='r', linestyle='--')
plt.text(x=avgd+2, y=h.max()*0.8, s='Average daily distance {:.1f} km'.format(avgd))
plt.title('User avg daily distance distribution')
plt.xlabel('Average daily distance [km]')
plt.ylabel('Frequency')    

