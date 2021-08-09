# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:36:59 2020

@author: U546416
"""

import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt

s=0.736
scale=np.exp(2.75)
loc=0

ln = stats.lognorm(s=s, scale=scale)
x = np.arange(0,100)

f, ax = plt.subplots(2)
plt.sca(ax[0])
plt.plot(x, ln.pdf(x), label='Borne')
plt.plot(x, stats.lognorm(s=s*0.8, scale=scale).pdf(x), label='s-20%')
plt.plot(x, stats.lognorm(s=s*1.2, scale=scale).pdf(x), label='s+20%')
plt.legend()
plt.sca(ax[1])
plt.plot(x, ln.pdf(x), label='Borne')
plt.plot(x, stats.lognorm(s=s, scale=scale*0.9).pdf(x), label='scale-20%')
plt.plot(x, stats.lognorm(s=s, scale=scale*1.1).pdf(x), label='scale+20%')
plt.legend()

print('Means:')
print(ln.mean())
print('s-20%', stats.lognorm(s=s*0.8, scale=scale).mean())
print('s+20%', stats.lognorm(s=s*1.2, scale=scale).mean())
print('scale-20%', stats.lognorm(s=s, scale=scale*0.8).mean())
print('scale+20%', stats.lognorm(s=s, scale=scale*1.2).mean())


#%%
img = []
for i in np.arange(0.5,1.5,0.1):
    row = []
    for j in np.arange(0.5,1.5,0.1):
        row.append(stats.lognorm(s=s*i, scale=scale*j).mean())
    img.append(row)

plt.imshow(img, origin='lower')
plt.xticks(np.arange(0.5,1.5,0.1))
plt.yticks(np.arange(0.5,1.5,0.1))
#%%
means = []

dx = np.arange(-10,11)

plt.plot(means)

#%% fitting

def logn(x, s, scale):
    return (1/(s*x/scale*np.sqrt(2*np.pi))*np.exp(-(np.log(x/scale)**2)/(2*s*s)))/scale

plt.figure()
plt.plot(x, ln.pdf(x))
plt.plot(x, logn(x, s, scale))

#%%    
from scipy.optimize import curve_fit
xdata = np.arange(2,100,5)
xdata_todistance = xdata*42/60
dx = 5*42/60
ydata = np.array([0.16,0.17,0.115,0.075,0.039,
         0.039,0.019,0.019,0.019,0.019,
         0.019,0.019,0.002,0.002,0.002,
         0.002,0.002,0.002,0.002,0.002])
ydata = ydata / ydata.sum()

# fit to distribution in distance
(sfit, scalefit), covs = curve_fit(f=logn, xdata=xdata_todistance, ydata=ydata/dx, p0=[s, scale], bounds=(-np.inf,np.inf))
# fit to distribution in trip duration
(sfit2, scalefit2), covs = curve_fit(f=logn, xdata=xdata, ydata=ydata/5, p0=[s, scale], bounds=(-np.inf,np.inf))

lnfit = stats.lognorm(s=sfit, scale=scalefit)
lnfit2 = stats.lognorm(s=sfit2, scale=scalefit2)

f, ax = plt.subplots(2)
plt.sca(ax[0])
plt.bar(xdata, ydata, width=5, alpha=0.3, label='VanRoy data')
plt.plot(x*60/42, lnfit.pdf(x)*dx, label='Fit to distance')
plt.plot(x, lnfit2.pdf(x)*5, label='Fit to duration')
plt.xlim([0,100])
plt.ylim([0,0.3])
plt.xticks([0,25,50,100])
plt.yticks(np.arange(0,0.3,0.05))
plt.grid()
plt.xlabel('Trip duration distribution [min]')
plt.title('Trip duration distribution, lognormal fitting')
plt.legend()

plt.sca(ax[1])
x2 = x*1.37
plt.plot(x2, lnfit.pdf(x)/1.37, label='Fit, mean={:.1f}'.format((lnfit.pdf(x)*x2).sum()))
plt.plot(x, ln.pdf(x), label='Borne, mean={:.1f}'.format((ln.pdf(x)*x).sum()))
plt.legend()
plt.xticks([0,25,50,100])
plt.xlim([0,100])
plt.ylim([0,0.06])
plt.yticks(np.arange(0,0.06,0.01))
plt.grid()
plt.xlabel('Distance distribution [km]')
plt.title('One-way trip distance, commuters [km]')