# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:12:46 2020

Useful functions to translate occupancy data files to charging schedules for EV model

@author: U546416
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_occupancy(file, idx_time='sec'):
    """ Reads a csv file with occupancy profiles.
    Returns a :
        pandas dataframe with occupancy profiles
        pandas series of ID of profile to Type of profile (FTE, PTE, School, Retired, Unemployed)
        pandas series of ID of profile to Household ID
    """
    occupancy = pd.read_csv(file, skiprows=2, index_col=0, low_memory=False)

    # Extracting type members on first row
    type_member = occupancy.loc['Member']
    # Computing household
    household = pd.Series(data=[int(i) for i in occupancy.columns.astype(float)], index=occupancy.columns)
    # removing that row from data
    occupancy.drop('Member', axis=0, inplace=True)
    occupancy = occupancy.iloc[:-1]
    # Transforming data type
    occupancy = occupancy.astype(float)
    occupancy.index = occupancy.index.astype(float)
    # datetime
    if idx_time == 'sec':
        dt = 60*60
    else:
        dt = 60
    day = pd.Series([int(i//(dt*24)) for i in occupancy.index], index=occupancy.index)
    hour = pd.Series([(i/(dt))%24 for i in occupancy.index], index=occupancy.index)
    
    # chaging index
    occupancy['day'] = day
    occupancy['hour'] = hour
    occupancy.set_index(['day', 'hour'], inplace=True)
    
    return occupancy, type_member, household

def get_users(type_member, users=None):
    """ Returns the users (columns of occupancy DF) of a given type.
    Default is all active users (FTE, PTE, Retired, Unemployed)
    """
    if users is None:
        users = ['FTE', 'PTE', 'Retired', 'Unemployed']
    return type_member[type_member.isin(users)].index 

def do_schedule(occupancy, type_member, verbose=False):
    # Arrival and departure matrix: +1 means departure (from home), -1 arrival, 0 change in state but always at home
    arrdep = ((occupancy == 3)*1).diff()
    
    # Creating df of 'ch. sessions'
    ndays = occupancy.index.get_level_values('day').max() + 1
    df = []
    # Iterating for each user (over columns)
    for i, series in arrdep.items():
        if verbose:
            if i%100 == 0:
                print('Computing user number {}'.format(i))
        # this k is to remove the changes in state but still at home
        k = series[~(series == 0)]
        # c is the counter of sessions
        c = 0
        utype = type_member[i]
        # Iterating for each occupancy event
        for t, j in k.items():
            # First change is the beginning that presents problems due to diff()
            if c == 0:
                c+=1
                continue
            # Get time id of event (start of session)
            inday = t[0]
            intime = t[1]
            # get time id of next event (end of session)
            try:
                outday = k.index[c+1][0]
                outtime = k.index[c+1][1]
            # if no next event bcs is the last one, check the first events
            except:
                # If the first event (ex. arrival at home) is the same than this one, then
                # the end of current event is at midnight day 0
                if k.iloc[1] == j:
                    p = 0
                # else, end of current event is at time of first event
                else:
                    p = 1
                outday = k.index[p][0] + ndays
                outtime = k.index[p][1]
            if j == -1:
                inout = True
            else:
                inout = False
            dt = (outday-inday)*24 + (outtime-intime)
            if dt < 0:
                dt += ndays*24
            if inday == 5:
                day = 'Sat'
            elif inday == 6:
                day = 'Sun'
            else:
                day = 'Weekday'
            df.append([i, utype, inout, inday, intime, outday, outtime, dt, day])
            c += 1
    
    cols = ['User', 'UserType', 'AtHome', 'ArrDay', 'ArrTime', 'DepDay', 'DepTime', 'DeltaTime', 'DoW']
    df = pd.DataFrame(df, columns=cols)
    return df

def do_charging_sessions(schedule):
    """
    """
    if not ('TripDist' in schedule):
        set_distances_to_schedule(schedule)
    chsessions = schedule[schedule.AtHome & ~(schedule.UserType == 'School')]
    chsessions = chsessions.drop(['DeltaTime', 'DoW', 'AtHome'], axis=1, inplace=False)
    return chsessions


def compute_trip_distances(schedule, dist_distrib=None,
                           avg_v=42, verbose=False):
    """
    """
    # Default option is use VanRoy
    if dist_distrib is None:
        # Lognormal distribution fitted to VanRoy data (in km per trip)
        dx = stats.lognorm(s=1.067,scale=8.543)
        dist_distrib = dict(Work=dict(distr=dx,
                                      scale=1.37),
                            Shopping=dict(distr=dx,
                                          scale=0.51),
                            Recreation=dict(distr=dx,
                                            scale=1),
                            Other=dict(distr=dx,
                                       scale=0.76))
    trips = schedule[~schedule.AtHome]
    workers = ['FTE', 'PTE']
    workerusers = schedule[schedule.UserType.isin(workers)].User.unique()
    distances = {}
    n=0
    if verbose:
        print('Computing work distances, nworkers {}'.format(len(workerusers)))
    for u in workerusers:
        if verbose:
            if n%100 == 0:
                print('\t{}'.format(n))
        n+=1
        # set distance to work
        dw=10000
        while dw > 120:
            dw = dist_distrib['Work']['distr'].rvs() * dist_distrib['Work']['scale']
        # Trip distance is 2 times dw (round trip)
        for i in schedule[(schedule.User == u) & (schedule.TripMotif == 'Work')].index:
            distances[i] = dw*2
        
    if verbose:
        print('Computing other trips, ntrips {}'.format(len(trips[~(trips.TripMotif == 'Work')])))
    for i, t in trips.iterrows():
        if t.TripMotif == 'Work':
            continue
        if verbose:
            if n%200 == 0:
                print('\t{}'.format(n))
        n+=1
        # each trip has a max distance limited by total duration away
        # we consider that at most a person will spend 2/3rd of time driving
        dmax = avg_v * t.DeltaTime/3
        d = 10000
        c=0
        while d > dmax:
            d = dist_distrib[t.TripMotif]['distr'].rvs() * dist_distrib[t.TripMotif]['scale']
            c +=1
            if c == 10:
                d=dmax*0.99
        distances[i] = d*2
    schedule.TripDistance = pd.Series(distances)
    return schedule

def define_trip_motif(schedule, vanroy=True): # , distance_distribs=None):
    workers = ['FTE', 'PTE']
    trips = schedule[~schedule.AtHome]
#    # Motif trip distance distribution
#    if distance_distribs is None:
#        distance_distribs = VanRoyDistribs
#    
    # motifs vector
    motifs = pd.Series(index=trips.index)
    # define work trips for workers
    mint = 3 # Min time to be considererd a work trip
    motifs[trips[(trips.UserType.isin(workers)) & (trips.DoW == 'Weekday') & (trips.DeltaTime >= mint)].index] = 'Work'
    
    if vanroy:
        # other motifs
        othermotifs = ['Shopping', 'Recreation', 'Other'] # i dont care about the others
        proba_motifs_wd = [0.348,0.344,0.308] #probability of each motif, normalized to sum 1
        proba_motifs_we = [0.367,0.417,0.216]
        for i, j in motifs[motifs.isnull()].iteritems():
            motifs[i] = np.random.choice(othermotifs, p=proba_motifs_wd if trips.DoW[i]=='Weekday' else proba_motifs_we)
    else:
        motifs[motifs.isnull()] = 'Other'
    return motifs

def set_distances_to_schedule(schedule, verbose=True):
    # get motifs:
    if verbose:
        print('Computing motifs')
    schedule['TripMotif'] = define_trip_motif(schedule, vanroy=True)
    if not ('TripDistance' in schedule):
        if verbose:
            print('Adding dummy row')
        schedule['TripDistance'] = 0
    if verbose:
        print('Computing distances')
    compute_trip_distances(schedule, verbose=verbose)
    return schedule

def plot_arr_dep_hist(hist, binsh=np.arange(0,24.5,0.5), ftitle=''):
    """ Plots arrival and departure histogram
    """
    f, (ax, ax2) = plt.subplots(1,2)
    i = ax.imshow(hist.T/hist.sum().sum(), origin='lower', extent=(0,24,0,24))
    ax.set_title('Distribution of sessions')
    ax.set_xlabel('Start of charging sessions')
    ax.set_ylabel('End of charging sessions')
    ax.set_xticks(np.arange(0,25,2))
    ax.set_yticks(np.arange(0,25,2))
    ax.set_xticklabels(np.arange(0,25,2))
    ax.set_yticklabels(np.arange(0,25,2))
    plt.colorbar(i, ax=ax)
    
    ax2.bar((binsh[:-1]+binsh[1:])/2, hist.sum(axis=1)/hist.sum().sum(), width=0.5, label='Arrivals')
    ax2.bar((binsh[:-1]+binsh[1:])/2, -hist.sum(axis=0)/hist.sum().sum(), width=0.5, label='Departures')
    ax2.set_xlim(0,24)
    ax2.set_xticks(np.arange(0,25,2))
    ax2.set_xticklabels(np.arange(0,25,2))
    ax2.set_title('Arrival and departure distribution')
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Distribution')
    ax2.legend()
    ax2.grid()
    f.suptitle(ftitle)
    f.set_size_inches(11.92,4.43)