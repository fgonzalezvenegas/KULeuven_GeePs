# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:20:04 2020
# Script that runs n households using occupancy(Continous script)
@author: U546416
"""
from occupancyContinuous import *
#from occupancy import *
import numpy as np 
import csv  
import os

cur_dir = os.getcwd()


# Number of buildings to generate (might get more disctint profiles)
nBui = 10
# Output folder and file with individual occupancy profiles
folder = r'c:\user\U546416\Documents\PhD\Data\KULeuven\\'
file = 'Occupancy_{}.csv'.format(nBui)


#%%
name = 'Household'
house='Household number'
member='Member'
dat=[]

# Creating occupancy profiles
for i in range(nBui):
    if i%10 == 0:
        print('Household {}'.format(i))
    hou = Household(str(name)+'_'+str(i), verbose=False)
    hou.simulate()
    var = np.array(hou.occ)
    var=np.insert(var,0,var[:,0],axis=1) # repeat first timestep for time 0
    mem = hou.members
    for m in mem:
        if m != 'U12': 
            member=np.append(member,m)
            house=np.append(house,i+1)
    if len(dat) != 0:
        dat = np.vstack((dat,var))
        
    else:
        dat = var

os.chdir(cur_dir)             
# and output the array to txt
print('writing')

# active -> 1 
# dat=np.where(dat==2, 0.5, dat) # sleeping -> 0.5 (was 1)
# dat=np.where(dat==3, 0, dat) # absent   -> 0 (was 3)

# time in minutes (seconds are stupid)
tim = np.arange(0,dat.shape[1]*10,10)
dataa = np.vstack((tim,dat))


with open(folder + file,'w',newline='') as fd: # command for Python 3  to avoid empty rows
#with open('Occupancy.csv','wb') as fd: # command for Python 2 
    nfw = csv.writer(fd)
#    nfw.writerow(['absent=0, sleeping=0.5, present&active=1'])
    nfw.writerow(['absent=3, sleeping=2, present&active=1'])
    nfw.writerow(['First column time in s'])
    nfw.writerow(house)
    nfw.writerow(member)
    nfw.writerows(dataa.T)
    