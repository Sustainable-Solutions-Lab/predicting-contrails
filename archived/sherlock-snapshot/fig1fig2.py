import warnings

warnings.filterwarnings("ignore")



# Classics

import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import geopandas as gpd

import os



# Others

from scipy.stats import gaussian_kde

from matplotlib.colors import LogNorm

from tqdm import tqdm



# Geometry and projection

from pyproj import Geod

from shapely.geometry import Point, LineString, MultiLineString

from shapely.ops import unary_union



#####################################################

################ UTILS ########################

########################################



import ephem

import math


df_full = pd.read_parquet("featuresdf.pq")

#df=df[:10000]

print(df_full)





x = np.sort(df_full.total_contrail_energy_forcing)[::-1]
n = df_full.shape[0]

# Create figure with broken axes
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6),
                              gridspec_kw={'width_ratios': [2, 1]})
plt.subplots_adjust(wspace=0.1)  # Reduce spacing between subplots

# Plot cooling segment (x < 0)
mask = np.where(x < 0)
ax1.plot(np.array([100 * k/n for k in range(n)])[mask],
        (100*np.cumsum(x)/np.sum(x))[mask],
        c='blue', lw=3, label='Cooling')
ax2.plot(np.array([100 * k/n for k in range(n)])[mask],
        (100*np.cumsum(x)/np.sum(x))[mask],
        c='blue', lw=3)

# Plot neutral segment (x == 0)
mask = np.where(x == 0)
ax1.plot(np.array([100 * k/n for k in range(n)])[mask],
        (100*np.cumsum(x)/np.sum(x))[mask],
        c='grey', lw=3, label='Neutral')
ax2.plot(np.array([100 * k/n for k in range(n)])[mask],
        (100*np.cumsum(x)/np.sum(x))[mask],
        c='grey', lw=3)

# Plot warming segment (x > 0)
mask = np.where(x > 0)
ax1.plot(np.array([100 * k/n for k in range(n)])[mask],
        (100*np.cumsum(x)/np.sum(x))[mask],
        c='red', lw=3, label='Warming')
ax2.plot(np.array([100 * k/n for k in range(n)])[mask],
        (100*np.cumsum(x)/np.sum(x))[mask],
        c='red', lw=3)

ax1.set_xlim(0, 20)  # outliers only
ax2.set_xlim(90, 100)  # most of the data
ax1.tick_params(right=False)  # Remove ticks on right side of left plot
ax2.tick_params(left=False)   # Remove ticks on left side of right plot

# Add break indicators
d = 0.03  # Slightly larger break markers
ax1.plot([1, 1 + d], [1, 1 - d], transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot([1, 1 + d], [0, d], transform=ax1.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [1, 1 - d], transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [0, d], transform=ax2.transAxes, color='k', clip_on=False)

# Hide spines between subplots
ax1.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)

# Horizontal reference lines (fixed to not cross breaks)
ax1.hlines(y=0, xmin=0, xmax=20, color='k', ls='--')
ax1.hlines(y=80, xmin=0, xmax=20, color='k', ls='--')
ax2.hlines(y=0, xmin=90, xmax=100, color='k', ls='--')
ax2.hlines(y=80, xmin=90, xmax=100, color='k', ls='--')

# Set custom x-ticks (5% increments)
ax1.set_xticks(np.arange(0, 21, 5))   # 0,5,10,15,20
ax2.set_xticks(np.arange(90, 101, 5)) # 90,95,100

# Labels and legend
ax1.set_ylabel('Cumulated % of the total RF')
fig.text(0.5, 0.04, '% of flights', ha='center', fontsize=12)
ax2.legend(loc='lower right'),# framealpha=0.7)  # Bottom right with transparency
# #Memory
x = None
plt.savefig('Lorentz_curve.eps', bbox_inches='tight')
plt.close()















#lorentz cure for flight distance

print("DISTANCE CURVE GENERATING")

df_long = df_full[df_full['total_flight_distance_km'].astype(float)>1000]

df_short = df_full[df_full['total_flight_distance_km'].astype(float)<1000]



print("dfs = long = ",df_long.shape, " df = short = ", df_short.shape)



# put the dfs into a list

dfs = [df_long,df_short]

# what style to use for each

styles = ['-','--']

# add to label

names = ['flightdist>1k_km','flightdist<1k_km']



# for each df, use xavier's trick
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6),
                              gridspec_kw={'width_ratios': [2, 1]})
plt.subplots_adjust(wspace=0.1)  # Reduce spacing between subplots
for i in range(len(dfs)):

        df = dfs[i]

        sty = styles[i]

        name = names[i]



        x = np.sort(df.total_contrail_energy_forcing)[::-1]



        print (" i = ",i, " x shape = ", x.shape, " x first = ", x[0:10], " last = ", x[:-10])



        print ("x.shape",x.shape)

        print ("x.shape[0]",x.shape[0])

        print ("making plot")
        # # < 0



        # # > 0

        ax1.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])]),

                 (100*np.cumsum(x)/np.sum(x)),

                 sty,

                 c = 'red',

                 lw = 3,

                 label = name)

        ax2.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])]),

                 ( 100*np.cumsum(x)/np.sum(x)),

                 sty,

                 c = 'red',

                 lw = 3,

                 label = name)



       
ax1.set_xlim(0, 20)  # outliers only
ax2.set_xlim(90, 100)  # most of the data
ax1.tick_params(right=False)  # Remove ticks on right side of left plot
ax2.tick_params(left=False)   # Remove ticks on left side of right plot

# Add break indicators
d = 0.03  # Slightly larger break markers
ax1.plot([1, 1 + d], [1, 1 - d], transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot([1, 1 + d], [0, d], transform=ax1.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [1, 1 - d], transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [0, d], transform=ax2.transAxes, color='k', clip_on=False)

# Hide spines between subplots
ax1.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)

# Horizontal reference lines (fixed to not cross breaks)
ax1.hlines(y=0, xmin=0, xmax=20, color='k', ls='--')
ax1.hlines(y=80, xmin=0, xmax=20, color='k', ls='--')
ax2.hlines(y=0, xmin=90, xmax=100, color='k', ls='--')
ax2.hlines(y=80, xmin=90, xmax=100, color='k', ls='--')

# Set custom x-ticks (5% increments)
ax1.set_xticks(np.arange(0, 21, 5))   # 0,5,10,15,20
ax2.set_xticks(np.arange(90, 101, 5)) # 90,95,100

# Labels and legend
ax1.set_ylabel('Cumulated % of the total RF')
fig.text(0.5, 0.04, '% of flights', ha='center', fontsize=12)
ax2.legend(loc='lower right'),# framealpha=0.7)  # Bottom right with transparency
# #Memory
x = None

plt.savefig('Lorentz_curve_distance_first.eps')

plt.close()





Season=[]
for l,therows in tqdm(df_full.iterrows()):
	thestring= therows['first_waypoint_time']
	month=thestring.month
	month=int(month)
	if month in {12,1,2}:
		Season.append('Winter')
	if month in {3,4,5}:
		Season.append("Spring")
	if month in {6,7,8}:
		Season.append("Summer")
	if month in {9,10,11}:
		Season.append("Fall")

df_full['Season']=Season

print(df_full['Season'])

#lorentz cure for seasons

print("season CURVE GENERATING")

df_winter = df_full[df_full['Season']=='Winter']
df_spring = df_full[df_full['Season']=='Spring']
df_summer = df_full[df_full['Season']=='Summer']
df_fall = df_full[df_full['Season']=='Fall']

# put the dfs into a list

dfs = [df_winter, df_spring, df_summer, df_fall]

# what style to use for each

styles = ['-','-','-','-']

# add to label

names = ['Winter','Spring','Summer','Fall']

colors=['#66c2a5','#abdda4','#d53e4f','#fdae61']

# for each df, use xavier's trick


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6),
                              gridspec_kw={'width_ratios': [2, 1]})
plt.subplots_adjust(wspace=0.1)  # Reduce spacing between subplots
for i in range(len(dfs)):

        df = dfs[i]

        sty = styles[i]

        name = names[i]

        color=colors[i]



        x = np.sort(df.total_contrail_energy_forcing)[::-1]



        print ("x.shape",x.shape)

        print ("x.shape[0]",x.shape[0])



        print ("making plot")



        # # < 0

        ax1.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])]),

                 ( 100*np.cumsum(x)/np.sum(x)),

                 sty,

                 c = color,

                 lw = 3,

                 label = name)
        ax2.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])]),

                 ( 100*np.cumsum(x)/np.sum(x)),

                 sty,

                 c = color,

                 lw = 3,

                label = name)


#save?



ax1.set_xlim(0, 20)  # outliers only
ax2.set_xlim(90, 100)  # most of the data
ax1.tick_params(right=False)  # Remove ticks on right side of left plot
ax2.tick_params(left=False)   # Remove ticks on left side of right plot

# Add break indicators
d = 0.03  # Slightly larger break markers
ax1.plot([1, 1 + d], [1, 1 - d], transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot([1, 1 + d], [0, d], transform=ax1.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [1, 1 - d], transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [0, d], transform=ax2.transAxes, color='k', clip_on=False)

# Hide spines between subplots
ax1.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)

# Horizontal reference lines (fixed to not cross breaks)
ax1.hlines(y=0, xmin=0, xmax=20, color='k', ls='--')
ax1.hlines(y=80, xmin=0, xmax=20, color='k', ls='--')
ax2.hlines(y=0, xmin=90, xmax=100, color='k', ls='--')
ax2.hlines(y=80, xmin=90, xmax=100, color='k', ls='--')

# Set custom x-ticks (5% increments)
ax1.set_xticks(np.arange(0, 21, 5))   # 0,5,10,15,20
ax2.set_xticks(np.arange(90, 101, 5)) # 90,95,100

# Labels and legend
ax1.set_ylabel('Cumulated % of the total RF')
fig.text(0.5, 0.04, '% of flights', ha='center', fontsize=12)
ax2.legend(loc='lower right')#, framealpha=0.7)  # Bottom right with transparency



plt.savefig('Lorentz_curve_seasons.eps')

plt.close()

# #Memory

x = None























# make one df for each curve we want, by applying restriction

# you could do this for any column in the df, here I just do this one as an example

df_Africa = df_full[df_full['Origin_Region']=='Africa']
df_Asia = df_full[df_full['Origin_Region']=='Asia']
df_Europe = df_full[df_full['Origin_Region']=='Europe']
df_MiddleEast =df_full[df_full['Origin_Region']=='Middle East']
df_NorthAmerica = df_full[df_full['Origin_Region']=='North America']
print("NorthAmerica: ",df_NorthAmerica.head())
df_Oceania = df_full[df_full['Origin_Region']=='Oceania']
df_Russia = df_full[df_full['Origin_Region']=='Russia']
df_SouthAmerica = df_full[df_full['Origin_Region']=='South America']





# put the dfs into a list

dfcontlist=[df_Asia, df_Europe, df_NorthAmerica]
# what style to use for each
styles = ['-','-','-']
# add to label
names = ['Asia', 'Europe', 'North America']
colors=['#d53e4f', '#fdae61','#abdda4']
# for each df, use xavier's trick
gap=.1
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6),
                              gridspec_kw={'width_ratios': [2, 1]})
plt.subplots_adjust(wspace=gap)  # Reduce spacing between subplots
for i in range(len(dfcontlist)):

        df = dfcontlist[i]

        sty = styles[i]

        name = names[i]

        color=colors[i]



        x = np.sort(df.total_contrail_energy_forcing)[::-1]



        print ("Continent: ",names[i])

        print ("x.shape",x.shape)

        print ("x.shape[0]",x.shape[0])



        print ("making plot")



        ax1.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])]),
                 ( 100*np.cumsum(x)/np.sum(x)),
                 sty,
                 c = color,
                 lw = 3,
                 label = name)

        ax2.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])]),
                 ( 100*np.cumsum(x)/np.sum(x)),
                 sty,
                 c = color,
                 lw = 3,
                 label = name)


ax1.set_xlim(0, 20)  # outliers only
ax2.set_xlim(90, 100)  # most of the data
ax1.tick_params(right=False)  # Remove ticks on right side of left plot
ax2.tick_params(left=False)   # Remove ticks on left side of right plot

# Add break indicators
d = 0.03  # Slightly larger break markers
ax1.plot([1, 1 + d], [1, 1 - d], transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot([1, 1 + d], [0, d], transform=ax1.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [1, 1 - d], transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot([-d, 0], [0, d], transform=ax2.transAxes, color='k', clip_on=False)

# Hide spines between subplots
ax1.spines.right.set_visible(False)
ax2.spines.left.set_visible(False)

# Horizontal reference lines (fixed to not cross breaks)
ax1.hlines(y=0, xmin=0, xmax=20, color='k', ls='--')
ax1.hlines(y=80, xmin=0, xmax=20, color='k', ls='--')
ax2.hlines(y=0, xmin=90, xmax=100, color='k', ls='--')
ax2.hlines(y=80, xmin=90, xmax=100, color='k', ls='--')


# Set custom x-ticks (5% increments)
ax1.set_xticks(np.arange(0, 21, 5))   # 0,5,10,15,20
ax2.set_xticks(np.arange(90, 101, 5)) # 90,95,100

# Labels and legend
ax1.set_ylabel('Cumulated % of the total RF')
fig.text(0.5, 0.04, '% of flights', ha='center', fontsize=12)
ax2.legend(loc='lower right')# framealpha=0.7)  # Bottom right with transparency

plt.savefig('Lorentz_curve_origonregion.eps',bbox_inches='tight', dpi=300)
plt.close()



