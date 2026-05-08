####
# This python file creates features and check some correlation from a cleaned Dataset
# We also need latitude and longitudes of Airports

# Remove warning (not mandatory)
import warnings

#from fig1fig2 import rf_filter
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
import xarray as xr
# Geometry and projection
from pyproj import Geod
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
import airportsdata
import seat_counts

from matplotlib import cm, colors
from datetime import timedelta

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


def sunradians(long, lat, time):
    # Update observer
    o.long = long * ephem.degree
    o.lat = lat * ephem.degree
    # Add the time (UTC)
    o.date = time
    # Compute the path
    sun.compute(o)
    # Compute the sinus only once
    return max(0, math.sin(sun.alt))


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


def is_night(long, lat, time):
    """
    Determine if it's night or day at a given location and time
    
    Parameters:
        - long, lat: longitude and latitude in degrees
        - time: datetime object (UTC)
    
    Returns:
        - 'night' if sun is below horizon, 'day' if sun is above horizon
    """
    # Update observer
    o.long = long * ephem.degree
    o.lat = lat * ephem.degree
    # Add the time (UTC)
    o.date = time
    # Compute the sun position
    sun.compute(o)
    # Check if sun is above horizon (altitude > 0)
    if sun.alt > 0:
        return 'day'
    else:
        return 'night'


def great_circle_coords(dep, arr, nb=20):
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
    l = np.array(geod.npts(dep[0], dep[1], arr[0], arr[1], nb))
    # Return coordinates
    return l


# We change the function so it return geometries directly
# It is necessary for the MultiLineString case
def great_circle_geometry(dep, arr, nb=20):
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
    arr = np.array(geod.npts(lon1=dep[0], lat1=dep[1], lon2=arr[0], lat2=arr[1], npts=nb, initial_idx=0, terminus_idx=0))

    # Check for flights that go over the antimeridian
    # Find the index where the sign changes in the first dimension
    sign_changes = np.where(np.diff(np.sign(arr[:, 0])))[0]

    if abs(min(arr[:,0]) - max(arr[:, 0])) > 180: #Other way around is faster
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


########################################
# Reading
# For silas
# Reading part
# For silas
# Reading part

import sys
from pathlib import Path
filter=sys.argv[1]
print("Reading files starting with "+filter)
data_dir = Path('Xavier_export_contrails')

for parquet_file in data_dir.glob(filter+"*-summary.pq"):
   print ("Reading file: "+parquet_file.name)

df = pd.concat(
   (pd.read_parquet(parquet_file)
   for parquet_file in data_dir.glob(filter+"*-summary.pq")),ignore_index=True)
#df = df.sample(n=20, random_state=42)

#df = pd.read_parquet("Xavier_export_contrails/20190101-summary.pq")

df = df.loc[~( (df["destination_airport"] == "") |
        (df["origin_airport"] == ""))
        ]

print('Reading ok. Columns are:')
print(df.columns)
useless_variable = None
df['seatcapacity'] = df['aircraft_type_icao'].apply(seat_counts.get_seat_capacity)
df['passengers']=df['seatcapacity']*df['load_factor']
print(df['seatcapacity'])
print(df['passengers'])
# constants
AGWP_100 = 82.5 * 10**(-15) # W/m²/kg
EI_CO2 = 3.16 # kgCO2/kg fuel
S_earth = 5.101 * 10**14 # m²

# Computing CO2 energy forcing
df['energy_forcing_CO2'] = AGWP_100 * 365 * 24 * 60 * 60 * df['total_fuel_burn'] * EI_CO2 * S_earth
df['contrail_index'] = df['total_contrail_energy_forcing'] / df['energy_forcing_CO2']
#df['contrail_CO2_km'] = df['total_contrail_energy_forcing'] / df['total_flight_distance_km']

df['contrail_CO2'] = df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth)
df['contrail_CO2_km'] = df['contrail_CO2'] / df['total_flight_distance_km']

#We define the impact of contrails per km
df['contrail_energy_forcing'] = df['total_contrail_energy_forcing'] / df['total_flight_distance_km']

df['contrail_RF'] = df['total_contrail_energy_forcing'] /  (365 * 24 * 60 * 60 * S_earth)
df['contrail_RF_km'] = df['contrail_RF'] / df['total_flight_distance_km']
df['Joulesperpasskm']=df['contrail_energy_forcing']/df['passengers']
useless_variable = None

####READ DATASET ERA5####


# Load ERA5 NetCDF for this filter/date
#era5_file = f"era5_pl_{filter[:-1]}.nc"
#print("ERA5 file:", era5_file)
#ds = xr.open_dataset(era5_file, engine="netcdf4")
####################
######## GEO COORDS
#####################

airports = airportsdata.load()

# First extract on all the airports present in the dataset
airport_loc = {}
missing_list = []
airport_names = list(airports.keys())

# Loop on airport code
for apt in tqdm(pd.Series(np.concatenate([
    df.origin_airport.values,
    df.destination_airport.values
])).unique()):

  if apt in airport_names:
    airport_loc[apt] = {
        'lon' : airports[apt]['lon'],
        'lat' : airports[apt]['lat'],
        ### Here add a Sjoin_nearest for the Region
    }
  else :
    missing_list.append(apt)
    #print(apt, 'not in airport list')
print("Missing airports: ",len(missing_list), missing_list)

# Memory
airports = None
airports_loc = None
# Origin
df['OriginLon'] = df.loc[~df.origin_airport.isin(missing_list)].origin_airport.apply(lambda x : airport_loc[x]['lon'])
df['OriginLat'] = df.loc[~df.origin_airport.isin(missing_list)].origin_airport.apply(lambda x : airport_loc[x]['lat'])


# Destination
df['DestinationLon'] = df.loc[~df.destination_airport.isin(missing_list)].destination_airport.apply(lambda x : airport_loc[x]['lon'])
df['DestinationLat'] = df.loc[~df.destination_airport.isin(missing_list)].destination_airport.apply(lambda x : airport_loc[x]['lat'])


# Drop missing airports
df.dropna(subset = ['OriginLon', 'DestinationLon'], inplace=True)
print("shape after airport clean", df.shape)

#########################################

"""# Aircraft performance database with cruise altitudes
#might need to add a few more
altitude_estimates = {
    # Airbus Family
    'A318': 39000, 'A319': 39000, 'A320': 39000, 'A321': 39000,
    'A21N': 39000, 'A20N': 39000, 'A19N': 39000,
    'A330': 42000, 'A332': 42000, 'A333': 42000, 'A33F': 42000, 'A339': 42000,
    'A342': 43000, 'A343': 43000, 'A345': 43000, 'A346': 43000,
    'A359': 43000, 'A35K': 43000, 'A388': 43000,
    'A300': 39000, 'A306': 39000, 'A310': 39000,
    
    # Boeing 737 Family
    'B731': 37000, 'B732': 37000, 'B733': 37000, 'B734': 37000,
    'B735': 37000, 'B736': 37000, 'B737': 37000, 'B738': 37000,
    'B739': 37000, 'B37M': 37000, 'B38M': 37000, 'B39M': 37000, 'B3JM': 37000,
    
    # Boeing 757/767
    'B752': 42000, 'B753': 42000,
    'B762': 42000, 'B763': 42000, 'B764': 42000,
    
    # Boeing 777 Family
    'B772': 43000, 'B77L': 43000, 'B773': 43000, 'B77W': 43000,
    'B778': 43000, 'B779': 43000, 'B77F': 43000,
    
    # Boeing 787 Family
    'B788': 43000, 'B789': 43000, 'B78J': 43000,
    
    # Boeing 747 Family
    'B741': 45000, 'B742': 45000, 'B743': 45000, 'B744': 45000,
    'B748': 45000, 'B74F': 45000, 'B748F': 45000,
    
    # Regional Jets
    'E135': 37000, 'E145': 37000, 'E170': 41000, 'E175': 41000,
    'E190': 41000, 'E195': 41000, 'E290': 41000, 'E295': 41000,
    'CRJ1': 41000, 'CRJ2': 41000, 'CRJ7': 41000, 'CRJ9': 41000, 'CRJX': 41000,
    
    # Turboprops
    'DH8A': 25000, 'DH8B': 25000, 'DH8C': 25000, 'DH8D': 25000,
    'AT42': 25000, 'AT43': 25000, 'AT72': 25000, 'AT76': 25000,
    'SF34': 25000, 'SH36': 25000,
    
    # Older Aircraft
    'DC10': 42000, 'MD11': 42000, 'MD80': 37000, 'MD82': 37000,
    'MD83': 37000, 'MD87': 37000, 'MD88': 37000, 'MD90': 37000,
    'L101': 42000,
    
    # Freighters
    'A306F': 39000, 'A30B': 39000, 'B763F': 42000, 'B744F': 45000,
    'MD11F': 42000, 'DC10F': 42000,
    
    # Business Jets
    'C25A': 45000, 'C25B': 45000, 'C25C': 45000, 'C56X': 45000, 'C680': 45000,
    'GLF4': 45000, 'GLF5': 45000, 'GL5T': 45000, 'GLEX': 45000,
    
    # Additional aircraft from warning
    'A148': 33000, 'A3ST': 30000, 'A400': 32000, 'AJ27': 40000, 'AJET': 42000,
    'ASTR': 40000, 'B462': 33000, 'B463': 33000, 'B712': 31000, 'B722': 31000,
    'BCS1': 39000, 'BCS3': 39000, 'BE40': 40000, 'BE4W': 40000, 'C17': 35000,
    'C500': 40000, 'C501': 40000, 'C510': 40000, 'C525': 40000, 'C550': 40000,
    'C551': 40000, 'C55B': 40000, 'C5M': 33000, 'C650': 40000, 'C700': 40000,
    'C750': 40000, 'CL30': 40000, 'CL35': 40000, 'CL60': 40000, 'DC87': 31000,
    'DC91': 31000, 'DC93': 31000, 'E35L': 40000, 'E45X': 40000, 'E50P': 40000,
    'E545': 40000, 'E550': 40000, 'E55P': 40000, 'E75L': 40000, 'E75S': 40000,
    'EA50': 30000, 'F100': 33000, 'F2TH': 40000, 'F70': 31000, 'F900': 40000,
    'FA10': 40000, 'FA20': 40000, 'FA50': 40000, 'FA7X': 45000, 'FA8X': 45000,
    'G150': 41000, 'G280': 41000, 'GA5C': 40000, 'GA6C': 40000, 'GALX': 45000,
    'GL7T': 45000, 'H25B': 40000, 'H25C': 40000, 'HA4T': 40000, 'HAWK': 40000,
    'HDJT': 40000, 'HUNT': 32000, 'J328': 30000, 'K35R': 37000, 'KC2': 42000,
    'L39': 32000, 'LJ31': 40000, 'LJ35': 40000, 'LJ40': 40000, 'LJ45': 40000,
    'LJ55': 40000, 'LJ60': 40000, 'LJ70': 40000, 'LJ75': 40000, 'MRF1': 40000,
    'PC24': 40000, 'PRM1': 40000, 'R135': 37000, 'R722': 37000, 'RJ1H': 33000,
    'RJ85': 33000, 'SBR1': 40000, 'SF50': 30000, 'SU95': 38000, 'T204': 40000,
    'T38': 42000, 'WW24': 40000
}

def feet_to_pressure_level(feet, available_levels):
    h = feet * 0.3048
    p0 = 1013.25  # hPa
    T0 = 288.15
    pressure_hPa = p0 * (1 - 0.0065 * h / T0) ** 5.257
    available_levels = np.array(available_levels)
    idx = (np.abs(available_levels - pressure_hPa)).argmin()
    return int(available_levels[idx])

ERA5_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 225, 200, 175, 150, 100, 70, 50]

df['pressure'] = df.apply(
    lambda row: feet_to_pressure_level(
        altitude_estimates.get(row.get('aircraft_type', ''), 37000) *
        (0.75 if row.get('flight_distance', 1000) < 300 else
         0.9 if row.get('flight_distance', 1000) < 800 else
         1.0 if row.get('flight_distance', 1000) < 2000 else
         1.05),
        ERA5_LEVELS
    ),
    axis=1
)
lontally=0
lattally = 0
print(df['pressure'])
def atmos_conditions_from_ds(ds, lat, lon, dt, pressure):
    # (function body as above, but take ds as input instead of filename)
    global lontally, lattally
    lon_grid = ds.longitude.values
    if lon < 0 and (lon_grid>180).any():
        lon = (lon + 360) % 360
    
    # Extract the nearest values using `.sel(..., method="nearest")`
    subset = ds.sel(
    valid_time=dt,
    pressure_level=pressure,
    latitude=lat,
    longitude=lon,
    method="nearest")
    actual_time = subset.valid_time.values
    actual_lat = subset.latitude.values
    actual_lon = subset.longitude.values
    if (lon - actual_lon)/lon > .1:
        lontally=lontally + 1    
    if (lat - actual_lat)/lat > .1:
        lattally=lattally + 1   
    pressure = str(int(pressure))
    ds_lev = ds.sel(valid_time=dt, pressure_level=int(pressure), method='nearest')
    pt = ds_lev.interp(latitude=lat, longitude=lon)
    temperature = subset['t'].item()  # scalar value
    u_wind = subset['u'].item()
    v_wind = subset['v'].item()
    geopotential = subset['z'].item()
    humidity = subset['r'].item()
    return temperature, humidity, u_wind, v_wind, geopotential"""


#####create seasons#####
# Ensure datetime
df['first_waypoint_time'] = pd.to_datetime(df['first_waypoint_time'])
month = df['first_waypoint_time'].dt.month

# Create northern hemisphere seasons (default)
df['Season'] = 'Autumn'  # Default
df.loc[month.isin([12, 1, 2]), 'Season'] = 'Winter'
df.loc[month.isin([3, 4, 5]), 'Season'] = 'Spring'
df.loc[month.isin([6, 7, 8]), 'Season'] = 'Summer'

# Create southern hemisphere seasons (opposite of northern)
df['Season_Southern'] = 'Autumn'  # Default
df.loc[month.isin([12, 1, 2]), 'Season_Southern'] = 'Summer'  # Opposite of Winter
df.loc[month.isin([3, 4, 5]), 'Season_Southern'] = 'Autumn'   # Opposite of Spring
df.loc[month.isin([6, 7, 8]), 'Season_Southern'] = 'Winter'   # Opposite of Summer
df.loc[month.isin([9, 10, 11]), 'Season_Southern'] = 'Spring' # Opposite of Autumn

# Determine which flights are in southern hemisphere based on origin/destination
# Consider a flight southern if either origin or destination is in southern hemisphere
southern_origin = df['OriginLat'] < 0
southern_dest = df['DestinationLat'] < 0
southern_flight = southern_origin | southern_dest

# Create a season column that uses southern hemisphere seasons for southern flights
df['Season_Weather'] = df['Season']  # Default to northern hemisphere
df.loc[southern_flight, 'Season_Weather'] = df.loc[southern_flight, 'Season_Southern']


print("seasons:", df['Season'][100:110])
print("southern flights:", southern_flight.sum(), "out of", len(df))
print("season_weather sample:", df['Season_Weather'][100:110])


####################
######## DAY/NIGHT CLASSIFICATION
#####################

# Create day/night classification based on sun position at origin airport at first waypoint time
print("Creating day/night classification...")
day_night_list = []

for lon, lat, time in tqdm(df[['OriginLon', 'OriginLat', 'first_waypoint_time']].values, 
                          desc='Day/Night classification'):
    day_night_list.append(is_night(lon, lat, time))

# Add the day/night column to the dataframe
df['day_night'] = day_night_list


# Print summary statistics
print("Day/Night distribution:")
print(df['day_night'].value_counts())
print(f"Percentage of night flights: {100 * (df['day_night'] == 'night').mean():.1f}%")


useless_variable = None 


useless_variable = None 


#############################################
## Overseas

# Create LineStrings geometries
lines = []

for k in tqdm(df.index, desc = 'Lines geometry'): # ~1h with 10 points
  # Get origin and destination
  dep = df.loc[k][[ 'OriginLon', 'OriginLat']].values
  arr = df.loc[k][[ 'DestinationLon', 'DestinationLat']].values

  #Compute geodesic flight path
  geo =  great_circle_geometry(dep, arr, nb = 10)

  # We need to separate in MultiLineString case the flights goes over the antimeridian
  lines.append(geo)


#Creating geodataframe
gdf = gpd.GeoDataFrame(df,
                       geometry = lines,
                       crs = 'epsg:4326'
                      )


# Memory
lines = None
useless_variable = None 


################################################################
#Regions attribution and related plots
world = gpd.read_file("ne_110m_admin_0_countries.zip")
print(world.columns)
print("world.CONTINENT #1")
print(world.CONTINENT)
#Putting french guyana in south america
world = pd.concat([world,
 gpd.GeoDataFrame(
                       data = {'NAME_EN' :[ 'French Guyana'],
                               'CONTINENT' : ['South America']},
                       geometry = [world[world.NAME_EN == 'France'].geometry.values[0].geoms[0]],
                       crs = 'epsg:4326'
                   )
                   ])
world.loc[world.NAME_EN == 'France', 'geometry'] = world[world.NAME_EN == 'France'].geometry.values[0].geoms[1:]


# Create Middle East
world.loc[
    ((world.NAME_EN == 'Saudi Arabia') |
    (world.NAME_EN == 'Yemen') |
    (world.NAME_EN == 'Afghanistan') |
    (world.NAME_EN == 'Oman') |
    (world.NAME_EN == 'Pakistan') |
    (world.NAME_EN == 'Syria') |
    (world.NAME_EN == 'Israel') |
    (world.NAME_EN == 'Lebanon') |
    (world.NAME_EN == 'Jordan') |
    (world.NAME_EN == 'Turkey') |
    (world.NAME_EN == 'Iran') |
    (world.NAME_EN == 'Iraq') |
    (world.NAME_EN == 'Qatar') |
    (world.NAME_EN == 'Palestine') |
    (world.NAME_EN == 'United Arab Emirates') |
    (world.NAME_EN == 'Cyprus') |
    (world.NAME_EN == 'N. Cyprus') |
    (world.NAME_EN == 'Kuwait') |
    (world.NAME_EN == 'Georgia') |
    (world.NAME_EN == 'Azerbaijan') |
    (world.NAME_EN == 'Armenia')
     ), 'CONTINENT'] = 'Middle East'


#Put Russia in it's own CONTINENT
world.loc[
     (world.NAME_EN == 'Russia') ,
     'CONTINENT'] = 'Russia'


world = world.loc[~ (world.CONTINENT == 'Seven seas (open ocean)')]
print("worLd.CONTINENT #2")
print(world.CONTINENT)


Origin = gpd.GeoDataFrame(index = df.index, geometry=gpd.points_from_xy(df['OriginLon'], df['OriginLat']))
Destination = gpd.GeoDataFrame(index = df.index, geometry=gpd.points_from_xy(df['DestinationLon'], df['DestinationLat']))
print("@@@@")
print(Origin)
print(Destination)
#Take some time as we have to match each
Origin = Origin.sjoin_nearest(world[['CONTINENT', 'geometry']])
Destination = Destination.sjoin_nearest(world[['CONTINENT', 'geometry']])
print("####")
print(Origin)
print(Destination)
print("duplication?")
print(Origin[Origin.index.duplicated()])
print(Destination[Destination.index.duplicated()])


Origin = Origin[~Origin.index.duplicated()]
Destination = Destination[~Destination.index.duplicated()]


print(Origin[Origin.index.duplicated()])
print(Destination[Destination.index.duplicated()])


df['Origin_Region'] = Origin['CONTINENT']
df['Destination_Region'] = Destination['CONTINENT']


df[(df.Origin_Region == 'Russia') & (df.Destination_Region == 'South_America')]
df[(df.Destination_Region == 'Russia') & (df.Origin_Region == 'South America')]#.total_flight_distance_km
df[[
     'origin_airport_name', 'origin_country',
       'destination_airport_name', 'destination_country',
     'Origin_Region', 'Destination_Region'
]]
print("CONTINENT values")


# Simple path geometry to check
l_geo = []

for orlon, orlat, deslon, deslat in tqdm(df[['OriginLon', 'OriginLat','DestinationLon', 'DestinationLat']].values):
  l_geo.append(LineString([(orlon, orlat), (deslon, deslat)]))


# Creation of geodataframe
gdf = gpd.GeoDataFrame(df, geometry = l_geo, crs = 'epsg:4326')
gdf['contrail_impact'] = gdf['total_contrail_energy_forcing'] / gdf['total_flight_distance_km']
gdf.sort_values(by = 'contrail_impact', ascending = False, inplace=True)


# This is useful to compute distance
geod = Geod(ellps = 'WGS84')


# Recompute the total distance according to our geometries
gdf['distance_km'] = gdf.geometry.apply(lambda x : geod.geometry_length(x) /1e3)



#### LAND score
##################################################################################

# # mask
world = unary_union(gpd.read_file("ne_110m_admin_0_countries.zip").geometry) # Simplify to make it faster?

# # Flights that overlapp the land
land = gdf.clip(mask = world) # ~Takes time due to the size (10min)

# # Compute the distance that overlapps
# # Takes some time too
land['distance_land_km'] = land.geometry.apply(lambda x : geod.geometry_length(x) /1e3)

# # How much of the flight was inland
land['land_score'] = land['distance_land_km'] / land['distance_km']

# print('We have a score > 100% for number of flights:', land[land.land_score > 1].shape[0])
# Normalize to 100%, theyre almost equal to 1 anyway
land.loc[land.land_score>1, 'land_score'] = 1

gdf['land_score'] = land['land_score']

# # Memory
# land = None
gdf['land_score'] = 100 * gdf.land_score.fillna(0)

############################################
######### Bearing #############

import math
#uses lat/long of each airport to calculate bearing of the flight, takes 1 minute
#assumes that flight is completely straight, although it probably averages out
def calc_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    return brng

Bearingcol=[]
# for i,row in tqdm(gdf.iterrows()):
#   lata=row['OriginLat']
#   latb=row['DestinationLat']
#   lona=row['OriginLon']
#   lonb=row['DestinationLon']
  ###
  # OLD - bearing=calc_bearing(lata,latb,lona,lonb)
  ###

# Faster - 10sec
for lona,lata,lonb,latb in tqdm(gdf[['OriginLon', 'OriginLat','DestinationLon', 'DestinationLat']].values):
  bearing=calc_bearing(lata,lona,latb,lonb)
  if bearing < 0:
    bearing=360+bearing
  Bearingcol.append(bearing)
gdf['bearing']=Bearingcol

################# NEW METHOD FOR ML NIGHTSCORE AND OTHER FEATURES #########################

# Decide the different shifts we want to have for teh night score (in hours)
list_shifts = [0, 1, 2, 3, 4, 5, 6]

from datetime import datetime


def calcweather(array_element, nb=30):
    lona, lata, lonb, latb, time1, time2, dist, RF = array_element
    times = pd.date_range(time1, time2)
    geo = great_circle_coords((lona, lata), (lonb, latb), nb=nb)
    
    results = [atmos_conditions_from_ds(ds, lon, lat, time, pressure)
               for lon, lat, time, pressure in zip(geo[:, 0], geo[:, 1], times, df['pressure'])]
    
    temps, humids, uwinds, vwinds, vspeeds = zip(*results)
    return np.mean(temps), np.mean(uwinds), np.mean(vwinds), np.mean(vspeeds)

def calctempimpact(array_element, nb=30):
    lona, lata, lonb, latb, time1, time2, dist, RF = array_element
    nb = max(1,round(dist/50))
    print("nb:",nb, "dist:", dist)
    # Compute geodesic flight path
    geo = great_circle_coords((lona, lata), (lonb, latb), nb=nb)
    #maybe remove some starting and ending points that aren't at altitude yet
    meanlat =  np.mean(geo[:, 1])
    maxlat = np.max(geo[:, 1])
    minlat = np.min(geo[:, 1])

    #RFmultiplier = -8.166e-07*pow(meanlat, 3) + 0.0001572*pow(meanlat, 2) - 0.0003607*(meanlat) + 0.4694
    coeffs = [-8.16645498e-07, 1.57186445e-04, -3.60734592e-04, 4.69374423e-01]
    poly_fit = np.poly1d(coeffs)
    KperRF = poly_fit(meanlat)
    maxlatKperRF = poly_fit(maxlat)


    return KperRF, maxlatKperRF, maxlat, minlat, meanlat

   
def calcnightscore(array_element, nb=30, list_shifts=list_shifts):
    lona, lata, lonb, latb, time1, time2, dist, RF = array_element
    
    # Compute geodesic flight path
    geo = great_circle_coords((lona, lata), (lonb, latb), nb=nb)
    
    to_return = list()
    to_returnb = list()
    
    for shift in list_shifts:
        # We shift the time
        time1_shift = time1 + timedelta(hours=shift)
        time2_shift = time2 + timedelta(hours=shift)
        
        # Time should be a list of times as the plane advance
        times = pd.date_range(time1_shift, time2_shift, periods=nb)
        
        # Format
        times = [x.strftime('%Y-%m-%d %H:%M:%S') for x in times]
        
        # Compute when the sun is up
        # Full values with average of the sinus
        sun_vals_full = [sunradians(lon, lat, time) for lon, lat, time in zip(geo[:, 0], geo[:, 1], times)]
        
        # Boolean indicator
        boolean = [1 if k > 0 else 0 for k in sun_vals_full]
        
        # # Adjusted only >0
        # adj_pos = [max(0, x) for x in sun_vals_full]
        # # Adjusted < 0
        # adj_neg = [min(0, x) for x in sun_vals_full]
        
        to_return.append(100 * (1 - np.mean(sun_vals_full)))
        to_returnb.append(100 * (1 - np.mean(boolean)))
    
    return to_return, to_returnb
####################
######## NIGHT SCORES
#####################

##########################################
# Compute on the entire dataframe

# Number of points to interpolate sun score
nb = 30

array = list(gdf[['OriginLon', 'OriginLat', 'DestinationLon', 'DestinationLat', 'first_waypoint_time', 'last_waypoint_time','total_flight_distance_km', 'contrail_RF_km']].values)

res = []
# Parallel iteration
s = time.time()

nightscores_full = []
nightscores_bool = []
calc_dist = []
tempimpacts =[]
temps = []
humids = []
uwinds = []
vwinds = []
vspeeds = []
maxlats=[]
minlats =[]
meanlats=[]
maxtempimpacts=[]
for l in list_shifts:
   nightscores_full.append([]) # make an empty list for each shift
   nightscores_bool.append([]) # make an empty list for each shift

geod = Geod(ellps='WGS84')

for flight in tqdm(array, desc='Shifted nightscores'):
    _,_,dist = geod.inv(flight[0],flight[1],flight[2],flight[3])
    calc_dist.append(dist/1000.)
   
   # calculate the full and bool nightscores from flight data
    resf,resb = calcnightscore(flight)
    #meantemp, meanhumid,meanuwind,meanvwind,meanvertv= calcweather(flight)
    tempimpact,maxtempimpact,maxlat,minlat, meanlat = calctempimpact(flight)
    tempimpacts.append(tempimpact)
    maxtempimpacts.append(maxtempimpact)
    maxlats.append(maxlat)
    minlats.append(minlat)
    meanlats.append(meanlat)


   # temps.append(meantemp)
   # humids.append(meanhumid)
   # uwinds.append(meanuwind)
   # vwinds.append(meanvwind)
   # vspeeds.append(meanvertv)   
    # add them to the columns we need for the dataframe
    for i in range(len(resf)):
        nightscores_full[i].append(resf[i])
    for i in range(len(resf)):
        nightscores_bool[i].append(resb[i])

# Add scores to the dataframe
for i in range(len(list_shifts)):
    shift = list_shifts[i]
    gdf['night_score_full_'+str(shift)] = nightscores_full[i]
    gdf['night_score_bool_'+str(shift)] = nightscores_bool[i]
gdf['Tempmultiplier']= tempimpacts
gdf['maxTempmultiplier']= maxtempimpacts
gdf['maxlat']= maxlats
gdf['minlat']= minlats
gdf['meanlat']=meanlats
gdf['speed']= gdf['total_flight_distance_km']/gdf['flight_duration_h']
gdf['total_flight_distance_km'] = calc_dist
print(".head:",gdf.head)
print("colummns:",gdf.columns)
print('Tempmultiplier:', gdf['Tempmultiplier'].head)
print('maxTempmultiplier:', gdf['maxTempmultiplier'].head)

print('OriginLat',gdf['OriginLat'])
print('destinLat',gdf['DestinationLat'])
print('maxlat',gdf['maxlat'])
print('minlat',gdf['minlat'])

#############################################
############## SAVING #######################

#gdf.to_parquet('/Users/silas/Documents/Watershed/testoutputfeatures.pq')

gdf.to_parquet('betterEFs/features_'+filter+ '_gdf.pq')



