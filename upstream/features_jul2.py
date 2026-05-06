####
# This python file creates features and check some correlation from a cleaned Dataset
# We also need latitude and longitudes of Airports

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
filter=sys.argv[1]
print("Reading files starting with "+filter)
from pathlib import Path
data_dir = Path('Xavier_export_contrails')

for parquet_file in data_dir.glob(filter+"*-summary.pq"):
   print ("Reading file: "+parquet_file.name)

df = pd.concat(
   (pd.read_parquet(parquet_file)
   for parquet_file in data_dir.glob(filter+"*-summary.pq")),ignore_index=True)

#df = pd.read_parquet("Xavier_export_contrails/20190101-summary.pq")
#df=df[:1000]

df = df.loc[~( (df["destination_airport"] == "") |
        (df["origin_airport"] == ""))
        ]

print('Reading ok. Columns are:')
print(df.columns)
useless_variable = None

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

useless_variable = None

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

################# NEW METHOD FOR ML NIGHTSCORE #########################

# Decide the different shifts we want to have for teh night score (in hours)
list_shifts = [0, 1, 2, 3, 4, 5, 6]

from datetime import datetime

def calcnightscore(array_element, nb=30, list_shifts=list_shifts):
    lona, lata, lonb, latb, time1, time2 = array_element
    
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
        
        to_return.append(100 * (np.mean(sun_vals_full)))
        to_returnb.append(100 * (1 - np.mean(boolean)))
    
    return to_return, to_returnb

####################
######## NIGHT SCORES
#####################

##########################################
# Compute on the entire dataframe

# Number of points to interpolate sun score
nb = 30

array = list(gdf[['OriginLon', 'OriginLat', 'DestinationLon', 'DestinationLat', 'first_waypoint_time', 'last_waypoint_time']].values)

res = []
# Parallel iteration
s = time.time()

nightscores_full = []
nightscores_bool = []
calc_dist = []

for l in list_shifts:
   nightscores_full.append([]) # make an empty list for each shift
   nightscores_bool.append([]) # make an empty list for each shift

geod = Geod(ellps='WGS84')

for flight in tqdm(array, desc='Shifted nightscores'):
   _,_,dist = geod.inv(flight[0],flight[1],flight[2],flight[3])
   calc_dist.append(dist/1000.)
   
   # calculate the full and bool nightscores from flight data
   resf,resb = calcnightscore(flight)
   #   print ("flight =", flight, 'res full = ',resf, ' res boolean = ',resb)
   
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

gdf['total_flight_distance_km'] = calc_dist
print(".head:",gdf.head)
print("colummns:",gdf.columns)

#############################################
############## SAVING #######################

#gdf.to_parquet('/Users/silas/Documents/Watershed/testoutputfeatures.pq')

gdf.to_parquet('features_'+filter+ '_gdf.pq')



