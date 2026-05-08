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













