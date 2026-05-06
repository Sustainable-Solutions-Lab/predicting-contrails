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
df = pd.read_parquet("/home/groups/sjdavis/silas/Watershed/betterEFs/")
# Read one file per season from Sherlockdfs
"""winter_file = "/home/groups/sjdavis/silas/Watershed/Sherlockdfs/features_2019011_gdf.pq"  # January (Winter)
spring_file = "/home/groups/sjdavis/silas/Watershed/Sherlockdfs/features_2019041_gdf.pq"  # April (Spring)
summer_file = "/home/groups/sjdavis/silas/Watershed/Sherlockdfs/features_2019071_gdf.pq"  # July (Summer)
autumn_file = "/home/groups/sjdavis/silas/Watershed/Sherlockdfs/features_2019101_gdf.pq"  # October (Autumn)
df_list = [
    pd.read_parquet(winter_file),
    pd.read_parquet(spring_file),
    pd.read_parquet(summer_file),
    pd.read_parquet(autumn_file)
]
df = pd.concat(df_list, ignore_index=True)"""
#print("df_list len:",len(df_list))
df = df.sample(n=10000000, random_state=42)
print("df random sample len:",len(df))
# If nb is not defined elsewhere, set a default value:
nb = 30 #number of points per flight

def great_circle_coords(dep, arr, nb):
    geod = Geod(ellps="WGS84")
    l = np.array(geod.npts(dep[0], dep[1], arr[0], arr[1], nb))
    return l
###GET POINTS###

trajx = np.array([])
trajy = np.array([])

sorteddf = df.sort_values(by=['tempimpact', 'total_flight_distance_km'], ascending=[True, True])
n = math.ceil(0.05 * len(sorteddf['tempimpact']))  # 5% index
threshold = sorteddf['tempimpact'].iloc[n-1]  # -1 for zero-based index
dfsubsetreverse = df[df['tempimpact'] <= threshold]

trajx_low, trajy_low = np.array([]), np.array([])
for row in tqdm(dfsubsetreverse[['OriginLon', 'OriginLat', 'DestinationLon', 'DestinationLat', 'first_waypoint_time', 'last_waypoint_time']].values, desc='5% lowest'):
    geo = great_circle_coords((row[0], row[1]), (row[2], row[3]), nb=nb)
    trajx_low = np.concatenate((trajx_low, geo[:, 0]))
    trajy_low = np.concatenate((trajy_low, geo[:, 1]))


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

# Set bright red for values above 12.5
custom_cmap.set_over('#d53e4f')  # Bright red
thres=math.ceil(len(df['destination_airport'])*.0004)
# Create normalization with custom range
norm = colors.Normalize(vmin=0, vmax=thres)

h, xedges, yedges, im=plt.hist2d(trajx_low,trajy_low,bins=[500,300],cmap=custom_cmap,cmin=1,norm=norm)

plt.ylim(-60,80)

# Add colorbar with cutoff indicator
cbar = plt.colorbar(im, extend='max', shrink=0.7)
cbar.set_label('5% most cooling flights at point')

plt.savefig('p2latratio_trajmapCooling10M.eps')
#plt.savefig('p2latratio_trajmapCooling10M').jpg')
plt.close()

