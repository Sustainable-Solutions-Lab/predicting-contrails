# Remove warning (not mandatory)
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
import airportsdata
from matplotlib import cm, colors
from datetime import date, datetime

#####################################################
################ UTILS ########################
########################################

import ephem
import math
import geodatasets
# Global sun variable
sun = ephem.Sun()
# Global observer
o = ephem.Observer()

AGWP_100 = 82.5 * 10**(-15) # W/m²/kg
EI_CO2 = 3.16 # kgCO2/kg fuel
S_earth = 5.101 * 10**14 # m² 
def sunradians(long, lat, time):
    # Update observer
   o.long = long * ephem.degree
   o.lat = lat * ephem.degree
  # Add the time (UTC)
   o.date = time
  # Compute the path
   sun.compute(o)
   # Compute the sinus only once
   return int(max(0, math.sin(sun.alt)))
def sunradiansfull(long, lat, time):
    # Update observer
   o.long = long * ephem.degree
   o.lat = lat * ephem.degree
  # Add the time (UTC)
   o.date = time
  # Compute the path
   sun.compute(o)
   # Compute the sinus only once
   return int(math.sin(sun.alt))
def sunradiansrev(long, lat, time):
  # Update observer
  o.long = long * ephem.degree
  o.lat = lat * ephem.degree
# Add the time (UTC)
  o.date = time
# Compute the path
  sun.compute(o)
# Compute the sinus only once
  return 1/int(math.sin(sun.alt))

def great_circle_coords(dep, arr, nb = 20):
    """Create the great circle geometry with pyproj
    parameters:
        - nb : number of points
        - dep, arr : departure and arrival
    return:
        - list of coordinates (lon, lat) of geodesic geometry
    """
    # projection
    geod = Geod(ellps="WGS84")
    # returns a list of longitude/latitude pairs describing npts equally spaced
    # intermediate points along the geodesic between the initial and terminus points.
    l = np.array(geod.npts(dep[0],
                           dep[1],
                           arr[0],
                           arr[1],
                           nb))
    # Return coordinates
    return l 

# We change the function so it return geometries directly
# It is necessary for the MultiLineString case

def great_circle_geometry(dep, arr, nb = 20):
    """Create the great circle geometry with pyproj
    parameters:
        - nb : number of points to interpolate trajectory
        - dep, arr : departure and arrival coordinates
    return:
        - shapely geometry
    """
    # projection
    geod = Geod(ellps="WGS84")
    # returns a list of longitude/latitude pairs describing npts equally spaced
    # intermediate points along the geodesic between the initial and terminus points.

    arr = np.array(geod.npts(
                      lon1 = dep[0],
                      lat1 = dep[1],
                      lon2 = arr[0],
                      lat2 = arr[1],
                      npts = nb,
                      # Necessary to go from start to end
                      initial_idx = 0,
                      terminus_idx = 0,))

    # Check for flights that go over the antimeridian
    # Find the index where the sign changes in the first dimension
    sign_changes = np.where(np.diff(np.sign(arr[:, 0])))[0]

    if abs(min(arr[:,0]) - max(arr[:, 0])) > 180: #Other way around is faster
    #sign_changes.size > 0:
        # Find the index where the sign changes in the first dimension
        sign_changes = np.where(np.diff(np.sign(arr[:, 0])))[0]
        idx = sign_changes[0]
        x1, y1 = arr[idx]
        x2, y2 = arr[idx + 1]

        # Interpolating the value at the zero-axis (180 or -180)
        # This will also prevent to have a 1-row array
        if x1 > 0 and x2 < 0:
            x_zero_1 = 180
            x_zero_2 = -180
            y_zero = y1 + (y2 - y1) * (x_zero_1 - x1) / abs(x2 - x1 + 360)
        elif x1 < 0 and x2 > 0:
            x_zero_1 = -180
            x_zero_2 = 180
            y_zero = y2 + (y1 - y2) * (x_zero_2 - x2) / abs(x1 - x2 + 360)

        # Splitting the array - it's more like first part / second part not positive / negative
        positive_part = np.vstack((arr[:idx + 1], [x_zero_1, y_zero]))
        negative_part = np.vstack(([x_zero_2, y_zero], arr[idx + 1:]))

        # Returning both geometries in a multilinestring
        return MultiLineString([LineString(positive_part), LineString(negative_part)])

    else:
        # Handle case where there is no sign change
        return LineString(arr)

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_

Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]


########################################
# Reading

# For silas
# Reading part
# Use the full dataset, not just a slice
# df = pd.read_parquet("/Users/silas/Documents/Watershed/Xavier_export_contrails/")#20190101-summary.pq")
# Try to read the parquet file - it might be a directory of parquet files
import glob
parquet_files = glob.glob("/Users/silas/Documents/Watershed/Xavier_export_contrails/*.pq")
if parquet_files:
    # Read the first file as a test
    df = pd.read_parquet(parquet_files[0])
    print(f"Loaded {parquet_files[0]}")
else:
    # Try reading as a directory
    df = pd.read_parquet("/Users/silas/Documents/Watershed/Xavier_export_contrails/")
    print("Loaded directory of parquet files")

# Ensure df is a DataFrame
print(f"Type of df: {type(df)}")
print(f"Shape of df: {df.shape if hasattr(df, 'shape') else 'No shape'}")
if not isinstance(df, pd.DataFrame):
    print("Warning: df is not a DataFrame, converting...")
    df = pd.DataFrame(df)
print(f"After conversion - Type of df: {type(df)}")
print(f"Columns: {df.columns.tolist() if hasattr(df, 'columns') else 'No columns'}")
df = df[:10000]
####################
######## GEO COORDS
#####################

# Load airport data
airports = airportsdata.load()
airport_names = set(airports.keys())

# Build airport location DataFrame for fast lookup
airport_loc_df = pd.DataFrame.from_dict(airports, orient='index')[['lon', 'lat']]
# Explicitly convert to Series and then to dict for mapping
lon_dict = pd.Series(airport_loc_df['lon'], index=airport_loc_df.index).to_dict()
lat_dict = pd.Series(airport_loc_df['lat'], index=airport_loc_df.index).to_dict()
df['OriginLon'] = df['origin_airport'].replace(lon_dict)
df['OriginLat'] = df['origin_airport'].replace(lat_dict)
df['DestinationLon'] = df['destination_airport'].replace(lon_dict)
df['DestinationLat'] = df['destination_airport'].replace(lat_dict)

# Drop missing airports
coord_cols = ['OriginLon', 'DestinationLon', 'OriginLat', 'DestinationLat']
df.dropna(subset=coord_cols, inplace=True)
print("shape after airport clean", df.shape)

############HUB AND SPOKE ANALYSIS###########

# Calculate contrail CO2 metrics
AGWP_100 = 82.5 * 10**(-15) # W/m²/kg
S_earth = 5.101 * 10**14 # m² 
df['contrail_CO2'] = df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth)
df['contrail_CO2_km'] = df['contrail_CO2'] / df['total_flight_distance_km']

# Top airports by destination count
count = pd.Series(df['destination_airport']).value_counts()
hub_airports = list(count.head(50).index)
print("top airports: ", hub_airports)

# Assign season using vectorized logic
# Ensure datetime
from pandas.api.types import is_datetime64_any_dtype
if not is_datetime64_any_dtype(df['first_waypoint_time']):
    df['first_waypoint_time'] = pd.to_datetime(df['first_waypoint_time'])
month = df['first_waypoint_time'].dt.month
conditions = [
    month.isin([12, 1, 2]).values,
    month.isin([3, 4, 5]).values,
    month.isin([6, 7, 8]).values,
    month.isin([9, 10, 11]).values
]
choices = ['Winter', 'Spring', 'Summer', 'Autumn']
df['Season'] = np.select(conditions, choices)
print("seasons:", df['Season'][100:110])

# Precompute per-airport and per-season means
# Only for hub airports
hub_df = df[df['destination_airport'].isin(hub_airports)]

# Group by airport and season
season_means = hub_df.groupby(['destination_airport', 'Season'])['contrail_CO2_km'].mean().unstack(fill_value=np.nan)
# Global mean per airport
global_means = hub_df.groupby('destination_airport')['contrail_CO2_km'].mean()
# Get coordinates for each airport
coords = hub_df.groupby('destination_airport')[['DestinationLon', 'DestinationLat']].first()

# Build lists for plotting (only airports with data)
airports_with_flights = list(global_means.index)
CF_mean = global_means.tolist()
airportx = coords['DestinationLon'].tolist()
airporty = coords['DestinationLat'].tolist()
airportc = global_means.tolist()

def safe_reindex(series, airports_with_flights):
    if series is None:
        return pd.Series([np.nan]*len(airports_with_flights), index=airports_with_flights)
    return series.reindex(airports_with_flights)

airportcW = safe_reindex(season_means.get('Winter', None), airports_with_flights).tolist()
airportcSp = safe_reindex(season_means.get('Spring', None), airports_with_flights).tolist()
airportcSu = safe_reindex(season_means.get('Summer', None), airports_with_flights).tolist()
airportcA = safe_reindex(season_means.get('Autumn', None), airports_with_flights).tolist()

print("CF list:", airportc)
print("Airports with flights:", airports_with_flights)
print("Number of airports with flights:", len(airports_with_flights))

# Plotting
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import matplotlib.pyplot as plt

# Bar plot
fig, ax = plt.subplots()
ax.bar(airports_with_flights, CF_mean)
plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
plt.ylabel("mean CF flights at given hub in KG CO2/km:")
plt.savefig("AI/p3bar.eps")
plt.savefig("AI/p3bar.jpg")
plt.close()

# Season-based maps
sznlist = ["W", "Sp", 'Su', "A", "global"]
sznsCF = [airportcW, airportcSp, airportcSu, airportcA, airportc]
for i, cf in enumerate(sznsCF):
    ax = gpd.read_file("ne_110m_admin_0_countries.zip").plot(color='white', edgecolor='black', lw=.5)
    hex_colors = [
        '#f46d43', '#fdae61', '#fee08b', 
        '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'
    ]
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', hex_colors[::-1], N=256)
    custom_cmap.set_over('#d53e4f')  # Bright red
    norm = colors.Normalize(vmin=0, vmax=12.5)
    scatter = plt.scatter(
        airportx, airporty,
        marker='o',
        c=cf,
        cmap=custom_cmap,
        norm=norm
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('CF in (KG CO₂)/Km per Flight')
    plt.xlim(-175, 175)
    plt.ylim(-60, 80)
    plt.savefig(f"AI/p3{sznlist[i]}.eps", dpi=300)
    plt.close()

# Location-based maps
ax = gpd.read_file("ne_110m_admin_0_countries.zip").plot(color='white', edgecolor='black', lw=.5)
hex_colors = [
    '#f46d43', '#fdae61', '#fee08b', 
    '#ffffbf', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'
]
custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', hex_colors[::-1], N=256)
custom_cmap.set_over('#d53e4f')  # Bright red
norm = colors.Normalize(vmin=0, vmax=12.5)
scatter = plt.scatter(
    airportx, airporty,
    marker='o',
    c=airportc,
    cmap=custom_cmap,
    norm=norm
)
cbar = plt.colorbar(scatter, extend='max')
cbar.set_label('CF in (KG CO₂)/Km per Flight')

# Add labels to each point
for i, (x, y, ap) in enumerate(zip(airportx, airporty, airports_with_flights)):
    x_offset = 2
    y_offset = 2
    if ap == 'KJFK':
        y_offset = 10
    plt.annotate(
        str(ap),
        (x, y),
        xytext=(x_offset, y_offset),
        textcoords='offset points',
        fontsize=8,
        ha='left',
        va='bottom'
    )
plt.xlim(-150, -50)
plt.ylim(15, 50)
plt.savefig('AI/p3USmap.eps', dpi=300)
plt.xlim(-10, 30)
plt.ylim(45, 55)
plt.savefig('AI/p3EUmap.eps', dpi=300)
plt.xlim(49, 152)
plt.ylim(-40, 50)
plt.savefig('AI/p3Asiamap.eps', dpi=300)
plt.close()




