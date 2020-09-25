# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:12:46 2020

Useful functions to translate occupancy data files to charging schedules for EV model

@author: U546416
"""
import pandas as pd
import numpy as np


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
    chsessions = schedule[schedule.AtHome & ~(schedule.UserType == 'School')]
    chsessions.drop(['DeltaTime', 'DoW', 'AtHome'], axis=1, inplace=True)
    if not ('TripDist' in schedule):
        compute_trip_distances(schedule)
    set_distances_to_charging_sessions(schedule)
    chsessions = schedule[schedule.AtHome & ~(schedule.UserType == 'School')]
    chsessions.drop(['DeltaTime', 'DoW', 'AtHome'], axis=1, inplace=True)
    return chsessions


def compute_trip_distances(schedule, work_dist, short_dist, other_dist):
    """
    """
    return schedule

def set_distances_to_charging_sessions(schedule):
    return schedule