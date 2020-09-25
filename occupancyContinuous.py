
import sys
import random
import numpy as np
import time
import datetime
import calendar
import os
cur_dir = os.getcwd()
strobeDir = r'C:\Users\u546416\AnacondaProjects\StROBe-master'
os.chdir(strobeDir + '\\Corpus\\')
import stats
import data

sys.path.append("..")
from Data.Households import households


class Household(object):
    '''
    The Household class is the main class of ProclivityPy, defining the
    household composition as input or randomized based on statistics and
    allowing simulation of the household for building energy simulations.

    Main functions are:
        - __init__(), which ...
        - self.parameterize(), which ...
        - self.simulate(), which ...
    '''

    def __init__(self, name, verbose=False, **kwargs):
        '''
        Initiation of Household object.
        '''
        # input ###############################################################
        # check on correct parameter input for use of functions as name should
        # be a string.
        try:
            if not isinstance(name, str):
                raise ValueError('Given name %d is not a string' % str(name))
        except:
            raise TypeError('give another name')
        # first define the name of the household object
        self.creation = time.asctime()
        self.name = name
        self.verbose = verbose
        self.parameterize(**kwargs)

    def parameterize(self, **kwargs):
        '''
        Get a household definition for occupants and present appliances based
        on average statistics or the given kwargs.
        '''

        def members(**kwargs):
            '''
            Define the employment type of all household members based on time
            use survey statistics or the given kwargs.
            '''
            members = []
            # First we check if membertypes are given as **kwargs
            if 'members' in kwargs:
                if isinstance(kwargs['members'], list):
                    members = kwargs['members']
                else:
                    raise TypeError('Given membertypes is no List of strings.')
            # If no types are given, random statististics are applied
            else:
                key = random.randint(1, len(households))
                members = households[key]
            # And return the members as list fo strings
            return members


        def clusters(members):
            '''
            Allocate each household member to the correct cluster based on the
            members occupation in time use survey data.
            '''
            clustersList = []
            # loop for every individual in the household
            for ind in members:
                if ind != 'U12':
                    clu_i = data.get_clusters(ind)
                    clustersList.append(clu_i)
            # and return the list of clusters
            return clustersList
        # and run all
        self.members = members(**kwargs)
        self.clustersList = clusters(self.members)
        # and return
        if self.verbose:
            print('Household-object created and parameterized.')
            print(' - Employment types are %s' % str(self.members))
            summary = [] #loop dics and remove dubbles
            for member in self.clustersList:
                summary += member.values()
            print(' - Set of clusters is %s' % str(list(set(summary))))
            
        return None

    def simulate(self, year=2013, ndays=365):
        '''
        The simulate function includes the simulation of the household
        occupancies, plug loads, lighting loads and hot water tappings.
        '''

        self.year = year
        self.__chronology__(year, ndays)
        self.__occupancy__()

    def __chronology__(self, year, ndays=None):
        '''
        A basic internal calendar is made, storing the days and months of the
        depicted simulating year.
        - Monday == 0 ... Sunday == 6
        '''
        # first we determine the first week of the depicted year
        fdoy = datetime.datetime(year,1,1).weekday()
        fweek = tuple(range(7)[fdoy:])
        # whereafter we fill the complete year
        if ndays:
            nday = ndays
        else:
            nday = 366 if calendar.isleap(year) else 365

        day_of_week = (fweek+tuple(range(7))*53)[:nday]
        # and return the day_of_week for the entire year
        self.dow = day_of_week
        self.nday = nday
        return None

    def __occupancy__(self, min_form = True, min_time = False):
        '''
        Simulation of a number of days based on cluster 'BxDict'.
        - Including weekend days,
        - starting from a regular monday at 4:00 AM.
        '''
        def check(occday, min_form = True, min_time = False): # -->> this check is not effective !!!
            '''
            We set a check which becomes True if the simulated day behaves
            according to the cluster, as a safety measure for impossible
            solutions. 
            ---->>> This check is not effectively implemented for the moment !!!
            '''

            # script 1 ########################################################
            # First we check if the simulated occ-chain has the same shape
            shape = True #  --->>> Check not effective, shape=True always !!!!!!!!!!!
            if min_form:
                location = np.zeros(1, dtype=int)
                reduction = occday[0]*np.ones(1, dtype=int)
                for i in range(len(occday)-1):
                    if occday[i+1] != occday[i]:
                        location = np.append(location, i+1)
                        reduction = np.append(reduction,occday[i+1])
#                shape = np.array_equal(reduction, RED) --->>> RED not available any more!!

            # script 2 ########################################################
            # And second we see if the chain has no sub-30 min differences
            length = True #  --->>> Check not effective, length=True always !!!!!!!!!!!
            if min_time:  # ------>>> min_time=False by default !!
                minlength = 99
                for i in location:
                    j = 0
                    while occday[i+j] == occday[i] and i+j < len(occday)-1:
                        j = j+1
                    if j < minlength:
                        minlength = j
                # and we neglect the very short presences of 20 min or less
                length = not minlength < 3

            # output ##########################################################
            # both have to be true to allow continuation, and we return boolean
            return shape and length # --->>> Check not effective, always True  !!!!!!!!!!!

        def dayrun(start, cluster):
            '''
            Simulation of a single day according to start state 'start'
            and the stochastics associated with 'cluster'.
            '''

            # script ##########################################################
            # we set the default daycheck at False for the first run and loop
            # creating days while False, meaning while the simulated day does
            # not correspond to the agreed-on rules in check().
            # ----->>> The check is currently not effectively implemented.
            daycheck = False
            end = datetime.datetime.utcnow() + datetime.timedelta(seconds = 10)
            # define the corresponding MCSA object from stats.py depicting a
            # Monte Carlo Survival Analysis object.
            SA = stats.MCSA(cluster)
            # and then keep simulating a day until True
            while daycheck == False:
                # set start state conditions
                tbin = 0
                occs = np.zeros(144, dtype=int)
                occs[0] = start
                # occupancy data from survey are given per 30 min, so we need to know in which of 48 bins to look for data:
                t48 = np.array(sorted(list(range(1, 49)) * 3))
                dt = SA.duration(start, t48[0]) # get duration of current state at start time (4am)
                # and loop sequentially transition and duration functions
                while tbin < 143:
                    tbin += 1
                    if dt == 0: # previous state duration has ended
                        occs[tbin] = SA.transition(occs[tbin-1], t48[tbin]) # find most probable next state
                        dt = SA.duration(occs[tbin], t48[tbin]) - 1 # restart duration counter for new state
                        # -1 is necessary, as the occupancy state already started
                    else:
                        occs[tbin] = occs[tbin - 1] # maintain current state
                        dt += -1 # count down duration of current state
                # whereafer we control if this day is ok
                daycheck = check(occs) # ----->>> The check is currently not effectively implemented, always TRUE! 
                # and we include a break if the while-loop takes to long until
                # check()-conditions are fulfilled.
                if datetime.datetime.utcnow() > end:
                    break

            # ouput ###########################################################
            # return occupants array if daycheck is ok
            return occs

        def merge(occ):
            '''
            Merge the occupancy profiles of all household members to a single
            profile denoting the most active state of all members.
            '''
            # scirpt ##########################################################
            # We start defining an array of correct length filled with the
            # least active state and loop to see if more-active people are
            # present at the depicted moment.
            occs = int(3)*np.ones(len(occ[0]))
            for member in occ:
                for to in range(len(member)):
                    if member[to] < occs[to]:
                        occs[to] = member[to]

            # ouput ###########################################################
            # return the merge occupancy states
            return occs

        # script ##############################################################
        # We change the directory to to location where the data is stored,
        # and run the three type of days, ie. wkdy, sat and son succesively
        # by which we can create a typical week.
        cdir = os.getcwd()
        occ_week = []
        for member in self.clustersList:
            startstate = 2 #sleeping @4.00 AM
            initday= dayrun(startstate, member['son']) # only to get new initial states for Monday
            mon = dayrun(initday[-1], member['wkdy']) # create new day for every weekday
            tue = dayrun(mon[-1], member['wkdy'])
            wed = dayrun(tue[-1], member['wkdy'])
            thu = dayrun(wed[-1], member['wkdy'])
            fri = dayrun(thu[-1], member['wkdy'])
            sat = dayrun(fri[-1], member['sat'])
            son = dayrun(sat[-1], member['son'])
            # and concatenate
            # week = np.concatenate((np.tile(wkdy, 5), sat, son, wkdy))
            week = np.hstack((initday, mon, tue, wed,thu,fri, sat, son))
            # bring last 4 h to the front so that data starts at midnight and delete extra day in front
            week = np.roll(week,24)
            week = week[144:]    
            occ_week.append(week)
            
#        # A merge occupancy is created depicted the most active state of all
#        # household members, later-on used for set-point temperatures and hot water tappings.
#        occ_merg = merge(occ_week)
#        # and combine the weekly occupancy states for the entire year by
#        # repeating them every week and correcting for the first day of year and stop time,
#        # including for the merged occupancy.
#        bins = 144
#        tstart = bins*self.dow[0]
#        tstop = tstart + bins*self.nday
#        occ_year = []
#        for line in range(len(occ_week)):
#            occ_year.append(np.tile(occ_week,54)[line][tstart:tstop])
#        occ_merged = []
#        occ_merged.append(np.tile(occ_merg,54)[tstart:tstop])


        # output ##############################################################
        # chdir back to original and return the occupancy states to the class
        # object.
        os.chdir(cdir)
        self.occ = occ_week
        self.occ_m = [merge(occ_week)]# self.occ_m = occ_merged
        # and print statements
        presence = [to for to in self.occ_m[0] if to < 2]
        hours = len(presence)/6.
        if self.verbose:
            print (' - Total presence time is {0:.1f} out of {1} hours'.format(hours, 7*24))
            print ('\tbeing {:.1f} percent)'.format(hours*100/(7*24)))
#        print (' - Total presence time is {0:.1f} out of {1} hours'.format(hours, 7*24))
#        print ('\tbeing {:.1f} percent)'.format(hours*100/(7*24)))
        return None
    



#name = 'Household'
#nBui = 1000
#path= r'C:\Users\u546416\AnacondaProjects\StROBe-master\\Corpus' 
#os.chdir(path)
#os.getcwd()
#
#house='Household number'
#member='Member'
#dat=[]
#
#for i in range(nBui):
#    if i%10 == 0:
#        print('Household {}'.format(i))
#    hou = Household(str(name)+'_'+str(i))
#    hou.simulate()
#    var = np.array(hou.occ)
#    var=np.insert(var,0,var[:,0],axis=1) # repeat first timestep for time 0
#    mem = hou.members
#    for m in mem:
#        if m != 'U12': 
#            member=np.append(member,m)
#            house=np.append(house,i+1)
#    if len(dat) != 0:
#        dat = np.vstack((dat,var))
#        
#    else:
#        dat = var
#             
########################################################################
## and output the array to txt
#print ('writing')
#
## active -> 1 
## dat=np.where(dat==2, 0.5, dat) # sleeping -> 0.5 (was 1)
## dat=np.where(dat==3, 0, dat) # absent   -> 0 (was 3)
#
#tim = np.linspace(0,604800,dat.shape[1])
#dataa = np.vstack((tim,dat))
#
#import csv  
#with open('Occupancy.csv','w',newline='') as fd: # command for Python 3  to avoid empty rows
##with open('Occupancy.csv','wb') as fd: # command for Python 2 
#    nfw = csv.writer(fd)
##    nfw.writerow(['absent=0, sleeping=0.5, present&active=1'])
#    nfw.writerow(['absent=3, sleeping=2, present&active=1'])
#    nfw.writerow(['First column time in s'])
#    nfw.writerow(house)
#    nfw.writerow(member)
#    nfw.writerows(dataa.T)
#    
#    
##%%    
###%% Script that runs n households
##
##
##name = 'Household'
##nBui = 1000
##path= r'C:\Users\u546416\AnacondaProjects\StROBe-master\Corpus' 
##os.chdir(path)
##os.getcwd()
##
##house='Household number'
##member='Member'
##dat=[]
##
##for i in range(nBui):
##    if i%10 == 0:
##        print('Household {}'.format(i))
##    hou = Household(str(name)+'_'+str(i), verbose=False)
##    hou.simulate()
##    var = np.array(hou.occ)
##    var=np.insert(var,0,var[:,0],axis=1) # repeat first timestep for time 0
##    mem = hou.members
##    for m in mem:
##        if m != 'U12': 
##            member=np.append(member,m)
##            house=np.append(house,i+1)
##    if len(dat) != 0:
##        dat = np.vstack((dat,var))
##        
##    else:
##        dat = var
##             
#########################################################################
### and output the array to txt
##print('writing')
##
### active -> 1 
### dat=np.where(dat==2, 0.5, dat) # sleeping -> 0.5 (was 1)
### dat=np.where(dat==3, 0, dat) # absent   -> 0 (was 3)
##
### time in minutes (seconds are stupid)
##tim = np.arange(0,dat.shape[1]*10,10)
##dataa = np.vstack((tim,dat))
##
##import csv  
##with open('Occupancy.csv','w',newline='') as fd: # command for Python 3  to avoid empty rows
###with open('Occupancy.csv','wb') as fd: # command for Python 2 
##    nfw = csv.writer(fd)
###    nfw.writerow(['absent=0, sleeping=0.5, present&active=1'])
##    nfw.writerow(['absent=3, sleeping=2, present&active=1'])
##    nfw.writerow(['First column time in s'])
##    nfw.writerow(house)
##    nfw.writerow(member)
##    nfw.writerows(dataa.T)
#    
#    
#    
#    