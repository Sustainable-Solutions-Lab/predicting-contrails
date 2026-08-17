import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from tqdm import tqdm
import math
from matplotlib import colors
from pyproj import Geod

# Load DataFrame from Sherlockdfs directory
#df = pd.read_parquet("/Users/silas/Documents/Watershed/Sherlockdfs/")
#df = pd.read_parquet("/home/groups/sjdavis/silas/Watershed/Sherlockdfs/")
# Read one file per season from Sherlockdfs
import sys
from pathlib import Path
columns_needed = [
    'Tempmultiplier', 
    'total_flight_distance_km',
    'OriginLon',
    'OriginLat', 
    'DestinationLon',
    'DestinationLat',
    'Joulesperpasskm',
   # 'preds',  # Used in commented sorting code
   # 'adjustedCO2kg/km'  # Used in commented cumulative calculation
]
filter=sys.argv[1]
if filter=='all':
    df = pd.read_parquet("/Users/silas/Documents/Watershed/adjustedEFs", columns=columns_needed)
    #df = df.sample(n=30000000, random_state=42)
    print('finished reading df')

if filter=="small":
    winter_file2 = "/Users/silas/Documents/Watershed/adjustedEFs/features_2021011_gdf.pq"  # January (Winter) /Users/silas/Documents/Watershed/adjustedEFs
    spring_file2 = "/Users/silas/Documents/Watershed/adjustedEFs/features_2021041_gdf.pq"  # April (Spring)
    summer_file2 = "/Users/silas/Documents/Watershed/adjustedEFs/features_2021071_gdf.pq"  # July (Summer)
    autumn_file2 = "/Users/silas/Documents/Watershed/adjustedEFs/features_2021101_gdf.pq"  # October (Autumn)
    df_list = [
        pd.read_parquet(winter_file2, engine='fastparquet'),
        pd.read_parquet(spring_file2, engine='fastparquet'),
        pd.read_parquet(summer_file2, engine='fastparquet'),
        pd.read_parquet(autumn_file2, engine='fastparquet')
    ]
    df = pd.concat(df_list)
    print("df len:",len(df))
    #df = df.sample(n=100000, random_state=42)
#print("df random sample len:",len(df))

# If nb is not defined elsewhere, set a default value:
nb = 30 #number of points per flight

def great_circle_coords(dep, arr, nb):
    geod = Geod(ellps="WGS84")
    l = np.array(geod.npts(dep[0], dep[1], arr[0], arr[1], nb))
    return l
###GET POINTS###
df['Joulesperpasskm'] = df['Joulesperpasskm'] * df['Tempmultiplier']

"""
sorteddf = df.sort_values(by = ['Joulesperpasskm', 'total_flight_distance_km'], ascending=[True, True])
n = math.ceil(0.95 * len(sorteddf['Joulesperpasskm']))
threshold=sorteddf['Joulesperpasskm'].iloc[n-1]
dfsubset= df[df['Joulesperpasskm'] >= threshold]

sorteddf = df.sort_values(by=['Joulesperpasskm', 'total_flight_distance_km'], ascending=[True, True])
n = math.ceil(0.05 * len(sorteddf['Joulesperpasskm']))  # 5% index
threshold = sorteddf['Joulesperpasskm'].iloc[n-1]  # -1 for zero-based index
dfsubsetreverse = df[df['Joulesperpasskm'] <= threshold]
"""

trajx_list = []
trajy_list = []
traj_RF_list = []

for row in tqdm(df[['OriginLon', 'OriginLat', 'DestinationLon', 'DestinationLat', 'Joulesperpasskm', 'total_flight_distance_km']].values):
    nb = max(2, round(row[5]/50))
    J = row[4]
    geo = great_circle_coords((row[0], row[1]), (row[2], row[3]), nb=30)
    
    trajx_list.append(geo[:, 0])
    trajy_list.append(geo[:, 1])
    traj_RF_list.append(np.full(geo.shape[0], J))

# Single concatenation at the end
trajx = np.concatenate(trajx_list)
trajy = np.concatenate(trajy_list)
traj_RF = np.concatenate(traj_RF_list)



# Compute sum of CO2 per bin and bin edges
sum_co2, xedges, yedges = np.histogram2d(trajx, trajy, bins=[360, 140], weights=traj_RF)
# Compute count of flights per bin (same as point count due to unique bins)
count_flights, _, _ = np.histogram2d(trajx, trajy, bins=[360, 140])

# Calculate average CO2, avoiding division by zero
average_RF = np.divide(sum_co2, count_flights, where=count_flights > 50)
average_RF[count_flights <= 50] = np.nan  # Mask bins with 5 or fewer flights
###CREATE MAP###
#World background
ax = gpd.read_file("ne_110m_admin_0_countries.zip").plot(color = 'white',
                                                                      edgecolor = 'black',
                                                                lw = .5)



# Create custom colormap from your hex values
from matplotlib.colors import LinearSegmentedColormap
hex_colors = [
    '#f46d43', '#fdae61', '#fee08b', 
    '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'
]
custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', hex_colors[::-1], N=256)
thres_RF = np.nanpercentile(average_RF, 98)  # Example: 95th percentile as threshold
custom_cmap.set_over('#d53e4f')  # Bright red

norm = colors.Normalize(vmin=0, vmax=thres_RF)
im = plt.pcolormesh(xedges, yedges, average_RF.T, 
                    cmap=custom_cmap, 
                    norm=norm, 
                    shading='auto')
# Set bright red for values above 12.5
# Create normalization with custom range






#h, xedges, yedges, im=plt.hist2d(trajx,trajy,bins=[140,360],cmap=custom_cmap,cmin=1,norm=norm) old code for reference


# Add colorbar with cutoff indicator
cbar = plt.colorbar(im, extend='max', shrink=0.7)
cbar.set_label('Mean Radiative Forcing (J/passengerkm)', fontsize=10)
cbar.ax.tick_params(labelsize=8)
plt.ylim(-60,80)
plt.savefig(filter+'warmingmap_Jan17.png', dpi=300, bbox_inches='tight')
plt.savefig(filter+'warmingmap_Jan17.eps', dpi=300, bbox_inches='tight')


#plt.savefig(filter+'warmingmap_aug7.png', dpi=300, bbox_inches='tight')

plt.close()
"""
if isinstance(df, pd.DataFrame):
    index = df.sort_values(by = ['preds', 'total_flight_distance_km'], ascending = [False, False]).index
    x = df.loc[index, 'adjustedCO2kg/km']
    x_axis_main = np.linspace(0, 100, x.size)
    y_axis_main = np.asarray(np.cumsum(x)/np.sum(x)*100)
    x_min = np.min(x_axis_main)
    x_max = np.max(x_axis_main)
    y_actual_min = np.min(y_axis_main)
    y_actual_max = np.max(y_axis_main)

    # Plot predictions
    ax1.plot(x_axis_main,
            y_axis_main,
            c='#f46d43', lw=3, label='Predictions')
    # Add 10% marker
    total_distance = x_axis_main[-1]
    ten_percent_distance = 0.1 * total_distance
    idx_10pct = np.searchsorted(x_axis_main, ten_percent_distance)
    x_10pct = x_axis_main[idx_10pct]
    y_10pct_cum = y_axis_main[idx_10pct]
    """