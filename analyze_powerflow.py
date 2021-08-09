# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:34:31 2020
Analyzing power flow results from KUL

@author: U546416
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading files
folder_res = r'c:\user\U546416\Documents\PhD\Data\KULeuven\ResultsPowerFlow\csv_from_python\\'
folder_figs = r'c:\user\U546416\Pictures\KU Leuven\PowerFlows\\'
#dumb = pd.read_csv(folder_res + 'R_rural_smart_false_rep_1_10_20201221.csv',
#                   engine='python')
#smart = pd.read_csv(folder_res + 'R_rural_smart_true_rep_1_10_20201221.csv',
#                   engine='python')
#smart10 = pd.read_csv(folder_res + 'Rural_smart_true_rep_11_20_20210205.csv',
#                   engine='python')
#dumb10 = pd.read_csv(folder_res + 'Rural_smart_false_rep_11_20_20210206.csv',
#                   engine='python')
dumb = pd.read_csv(folder_res + 'R_rural_smart_false_reps1-20.csv',
                   engine='python')
smart = pd.read_csv(folder_res + 'R_rural_smart_true_reps1-20.csv',
                   engine='python')

# columns: Columns 0-8 are indexes of reinforcemnet options
#'Unnamed: 0', '01_Nom_Pow_Tra', '02_Max_Line_Cur', '03_nPhase',
#'04b_Ndum', '05_HP_Impl_Rate', '06_PV_Impl_Rate', '06b_EV_Impl_Rate',
#'07_nline_InputsF', '08_rep', '09_V_Dev', '10_VUF', '11_Tra_use_fac',
#'12_Cable_use', '14_PWRRi', '14_PWRRi_h', '16_Inv_tra', '17_Inv_cable',
#'18_Inv_1_to_3_phase', '18b_HH_1_to_3_phase', '19_Inv_total',
#'26_cost_ohmic_loss', '30_avg_PtraTot', '33_avg_feeder_PHP',
#'36_avg_feeder_Pplug', '39_avg_feeder_PPVFinal', '39b_avg_feeder_PEV',
#'45_avg_dummy_PHP', '48_avg_dummy_Pplug', '51_avg_dummy_PPVFinal',
#'51b_avg_dummy_PEV', '55b_maxIh', '55c_maxIdl_h'

cols = ['ID', 'TrafokW', 'FeederAmp', 'nPhase', 'Ndumb', 'HP_rate', 'PV_rate', 'EV_rate',
                'nline_InputsF', 'rep', 'V_dev', 'VUF', 'Trafo_ovl','Cable_ovl', 
                'PWRRi', 'PWRRi_h', 'Inv_tra', 'Inv_cable','Inv_1_to_3_phase', 'HH_1_to_3_phase', 'Inv_total',
                'Losses_cost', 'PtraTot_avg', 'Feeder_PHP_avg', 'Feeder_Pplug_avg', 'Feeder_PPVFinal_avg', 'Feeder_PEV_avg',
                'Dummy_PHP_avg', 'Dummy_Pplug_avg', 'Dummy_PPVFinal_avg', 'Dummy_PEV_avg', 'Ih_max', 'Idl_h_max']
dumb.columns = cols
smart.columns  = cols
#dumb10.columns = cols
#smart10.columns  = cols
#
#dumb = pd.concat([dumb, dumb10], ignore_index=True)
#smart = pd.concat([smart, smart10], ignore_index=True)
#
#dumb.to_csv(folder_res + 'R_rural_smart_false_reps1-20.csv', index=False)
#smart.to_csv(folder_res + 'R_rural_smart_true_reps1-20.csv', index=False)

dumb.set_index(['TrafokW', 'FeederAmp', 'nPhase', 'HP_rate', 'PV_rate', 'EV_rate', 'rep'], inplace=True)
smart.set_index(['TrafokW', 'FeederAmp', 'nPhase', 'HP_rate', 'PV_rate', 'EV_rate', 'rep'], inplace=True)

#
def remove_extremes(data, indicator='V_dev'):

    drops = []
    for h in data.index.levels[0]:
        for p in data.index.levels[1]:
            for e in data.index.levels[2]:
                drops.append((h,p,e,data[indicator].loc[h,p,e,:].idxmax()))
                drops.append((h,p,e,data[indicator].loc[h,p,e,:].idxmin()))
    return data.drop(drops)


def plot_indicator(df, indicator='V_dev', ax=None, 
                   hplevels=[0], pvlevels=[0], 
                   namelabel='', plotparams=dict(),
                   errorbars=True, legend=True):
    if ax is None:
        f, ax = plt.subplots()
    for hpl in hplevels:
        for pvl in pvlevels:
            avg = df.loc[hpl, pvl].groupby('EV_rate')[indicator].mean()
            if errorbars:
                lowerr = (avg-df.loc[hpl, pvl].groupby('EV_rate')[indicator].min()).values
                higherr = (df.loc[hpl, pvl].groupby('EV_rate')[indicator].max()-avg).values
                ax.errorbar(avg.index, avg, yerr=np.array([lowerr, higherr]), 
                            label=namelabel + 'HP {}%; PV {}%'.format(hpl, pvl), 
                            **plotparams)
            else:
                ax.plot(avg.index, avg, 
                        label=namelabel + 'HP {}%; PV {}%'.format(hpl, pvl), 
                            **plotparams)   
            plt.xlabel('EV integration [%]')
            plt.ylabel(ind)
    if legend:
        ax.legend()
    return ax


# Get data for base case
trafo0 = 160000
feeder0 = 225
nphase0 = 1

dumb0 = dumb.loc[trafo0, feeder0, nphase0, :,:,:,:]
smart0 = smart.loc[trafo0, feeder0, nphase0, :,:,:,:]

dumb_avg = dumb.groupby(['TrafokW', 'FeederAmp', 'nPhase', 'HP_rate', 'PV_rate', 'EV_rate']).mean()
smart_avg = smart.groupby(['TrafokW', 'FeederAmp', 'nPhase', 'HP_rate', 'PV_rate', 'EV_rate']).mean()


#%% Plot avg indicators for base case
#
for ind in ['V_dev', 'VUF', 'Trafo_ovl','Cable_ovl']:
    plotparamsd=dict(elinewidth=0.5, capsize=2)
    plotparamss=dict(elinewidth=0.5, capsize=2,linestyle='--', uplims=True, lolims=True)
    f, ax = plt.subplots()
    ax.set_title(ind)
    plot_indicator(dumb0, ax=ax, indicator=ind, 
                   hplevels=[0,100], pvlevels=[0,100], 
                   namelabel='Dumb ',
                   plotparams=plotparamsd)
    plot_indicator(smart0, ax=ax, indicator=ind, 
                   hplevels=[0,100], pvlevels=[0,100], namelabel='Smart ',
                   plotparams=plotparamss)
#%% Plot average indicators, but removing extreme cases


for ind in ['V_dev', 'VUF', 'Trafo_ovl','Cable_ovl']:
    plotparamsd=dict(elinewidth=0.5, capsize=2)
    plotparamss=dict(elinewidth=0.5, capsize=2,linestyle='--', uplims=True, lolims=True)
    f, ax = plt.subplots()
    ax.set_title(ind)
    dumb_red = remove_extremes(dumb0, indicator=ind)
    smart_red = remove_extremes(smart0, indicator=ind)
    plot_indicator(dumb_red, ax=ax, indicator=ind, 
                   hplevels=[0,100], pvlevels=[0,100], 
                   namelabel='Dumb ',
                   plotparams=plotparamsd)
    plot_indicator(smart_red, ax=ax, indicator=ind, 
                   hplevels=[0,100], pvlevels=[0,100], namelabel='Smart ',
                   plotparams=plotparamss)    

#%% Plot for paper, 4 indicators in one plot
f, axs = plt.subplots(2,2)
errorbars = False

labels= ['Transfo. Ovl. [%]', 'Ovl. segments [%]', 'V. deviation [%]', 'V. Unbalances [%]']
for j, ind in enumerate(['Trafo_ovl','Cable_ovl','V_dev', 'VUF']):
    ax = axs[j//2][j%2]
    plt.sca(ax)
    plotparamsd = dict()
    plotparamss = dict(linestyle='--')
    if errorbars:
        plotparamsd=dict(elinewidth=0.5, capsize=2)
        plotparamss=dict(elinewidth=0.5, capsize=2,linestyle='--', uplims=True, lolims=True)
    dumb_red = remove_extremes(dumb0, indicator=ind)*100
    smart_red = remove_extremes(smart0, indicator=ind)*100
    plot_indicator(dumb_red, ax=ax, indicator=ind, 
                   hplevels=[0,100], pvlevels=[0,100], namelabel='Unc.; ',
                   plotparams=plotparamsd, legend=False, errorbars=errorbars)  
    plot_indicator(smart_red, ax=ax, indicator=ind, 
                   hplevels=[0,100], pvlevels=[0,100], namelabel='Smart; ',
                   plotparams=plotparamss, legend=False, errorbars=errorbars)  
    ax.set_ylabel(labels[j])

f.set_size_inches(9.4 , 4.84)
f.tight_layout()

f.legend(ncol=4, loc=8)
# resizing axs to leave space for legend
for i, axx in enumerate(axs):
    for j, ax in enumerate(axx):
        pos = ax.get_position()
        dy = 0.04
        ax.set_position([pos.x0, pos.y0+dy*(i+1), pos.width, pos.height-dy+0.01])
        
nreps = dumb0.index.levels[-1].max()
f.savefig(folder_figs + 'grid_stability_{}reps{}.png'.format(nreps, '_err' if errorbars else ''))
f.savefig(folder_figs + 'grid_stability_{}reps{}.pdf'.format(nreps, '_err' if errorbars else ''))

    
#%% 
def get_min_inv(df, thresholds=dict(VUF=0, V_dev=0, Trafo_ovl=0.1, Cable_ovl=0)):
    df_avg = df.groupby(['TrafokW', 'FeederAmp', 'nPhase', 'HP_rate', 'PV_rate', 'EV_rate']).mean()
    df_avg = df_avg.reset_index(drop=False)
    hplevels = df_avg.HP_rate.unique()
    pvlevels = df_avg.PV_rate.unique()
    evlevels = df_avg.EV_rate.unique()
    idxs = []
    for hp in hplevels:
        for pv in pvlevels:
            for ev in evlevels:
                subdf = df_avg[(df_avg.HP_rate==hp) & (df_avg.PV_rate==pv) & (df_avg.EV_rate==ev)]
                # Remove 3phase case when there is no DERs
                if hp==0 & pv==0 & ev==0:
                    subdf[subdf.nPhase == 1]
                # Select only viable cases based on thresholds for each indicator
                for ind, thr in thresholds.items():
                    subdf = subdf[subdf[ind]<=thr]
                idx = subdf.PWRRi_h.idxmin()
                idxs.append(idx)
    df_avg = df_avg.loc[idxs]
    df_avg.set_index(['HP_rate', 'PV_rate', 'EV_rate'], inplace=True)
    return df_avg

mininv_dumb = get_min_inv(dumb, thresholds=dict(VUF=0, V_dev=0, Trafo_ovl=0., Cable_ovl=0))
mininv_smart = get_min_inv(smart, thresholds=dict(VUF=0, V_dev=0, Trafo_ovl=0., Cable_ovl=0))




#%%
def plot_investment(df,  ax=None, hplevels=0, pvlevels=0, namelabel='', plotparams=dict()):
    if ax is None:
        f, ax = plt.subplots()
    data = df.loc[hplevels, pvlevels,:]['Inv_tra', 'Inv_cable', 'Inv_1_to_3_phase']
    ax.stackplot(data.index, , 
                **plotparams)
    plt.xlabel('EV integration [%]')
    plt.ylabel(ind) 
    ax.legend()
    return ax