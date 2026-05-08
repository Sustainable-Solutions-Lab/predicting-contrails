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
########################################
# Reading

# For silas
# Reading part
# For silas
# Reading part

"""import sys
filter=sys.argv[1]
print("Reading files starting with "+filter)
from pathlib import Path
data_dir = Path('Xavier_export_contrails')
df = pd.concat(
   (pd.read_parquet(parquet_file)
   for parquet_file in data_dir.glob(filter+"*-summary.pq")),ignore_index=True)"""

df = pd.read_parquet("/home/groups/sjdavis/silas/Watershed/Xavier_export_contrails/20190101-summary.pq")
#df=df[:1000]











df = df.loc[~( (df["destination_airport"] == "") |
        (df["origin_airport"] == ""))
        ]

print('Reading ok')
print(df.index)
print(df.columns)
useless_variable = None 
####################
# FIGURES
################################################

### CO2

plt.hist(df['total_fuel_burn'] /( df['total_flight_distance_km'] ), bins = 50)

plt.xlabel('Consumption ? --> (kg fuel/km) probably')

plt.savefig('fuel_consumption.png')
plt.close()


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

# df['contrail_CO2'] = df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth)

fig, ax = plt.subplots(1, 3, figsize = (12,4))

ax[0].hist(df['total_fuel_burn'] * EI_CO2  / df['total_flight_distance_km'], bins = 50, range=(0, 100))
ax[0].set_xlabel('CO2 emissions (kg/km)')

ax[1].hist(df['total_contrail_energy_forcing'] / (AGWP_100 * 365 * 24 * 60 * 60 * df['total_fuel_burn'] * EI_CO2 * S_earth), bins = 50, range=(-2, 5))
ax[1].set_xlabel('Contrail index (ratio)')

ax[2].hist(df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth * df['total_flight_distance_km']), bins = 50, range=(-100, 500))
ax[2].set_xlabel('Contrail CO2 (kg/km)')

fig.tight_layout()

plt.savefig('CO2_and_contrails.png')
plt.close()

# Time scope
plt.figure(figsize = (10, 5))
plt.hist(df.first_waypoint_time, bins = 50)
plt.ylabel('Number of flights')
plt.xlabel('Dates')
plt.savefig('Time_scope.png')
plt.close()


# Pie chart

labels = 'No Contrails (RF = 0)', 'RF > 0', 'RF < 0'
#sizes = [
#         100 * df[df.total_contrail_energy_forcing == 0].shape[0]/df.shape[0],
#         100 * df[df.total_contrail_energy_forcing > 0].shape[0]/df.shape[0],
#         100 * df[df.total_contrail_energy_forcing < 0].shape[0]/df.shape[0]]

#fig, ax = plt.subplots()
#ax.pie(sizes, labels=labels, autopct='%1.2f%%')
#plt.savefig('Pie_chart.png')
#plt.close()

x = np.sort(df.total_contrail_energy_forcing)

 # # < 0
plt.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])])[np.where(x < 0)],
        ( 100*np.cumsum(x)/np.sum(x))[np.where(x < 0)],
        c = 'blue',
        lw = 3,
        label = 'Cooling')

# # = 0
plt.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])])[np.where(x == 0)],
         (100*np.cumsum(x)/np.sum(x))[np.where(x == 0)],
         c = 'grey',
         lw = 3,
         label = 'Neutral')

# # > 0
plt.plot(np.array([100 * k/df.shape[0] for k in range(df.shape[0])])[np.where(x > 0)],
         (100*np.cumsum(x)/np.sum(x))[np.where(x > 0)],
         c = 'red',
         lw = 3,
         label = 'Warming')


plt.xlabel('% of flights')
plt.ylabel('Cumulated % of the total RF')
plt.hlines(0, 0, 100,
           color = 'k',
           ls = '--')

plt.hlines(20, 0, 100,
           color = 'k',
           ls = '--')

plt.legend()
# #Memory
x = None

plt.savefig('Lorentz_curve.png')
plt.close()



plt.hist(df['total_contrail_energy_forcing'],
         bins = 100,
        # range = (-1e14, 1e14)
         )
plt.yscale('log')
plt.ylabel('Flights (log scale)')
plt.xlabel('Total contrail energy forcing (Joules)')

plt.savefig('Histogram.png')
plt.close()


print('First figures okay')
###############################################################

#We define the impact of contrails per km
df['contrail_energy_forcing'] = df['total_contrail_energy_forcing'] / df['total_flight_distance_km']

plt.hist(df['contrail_energy_forcing'],
         bins = 100,
        # range = (-1e14, 1e14)
         )
plt.yscale('log')
plt.ylabel('Flights (log scale)')
plt.xlabel('Contrail energy forcing (Joules/km)')

plt.savefig('Histogram_normalized.png')
plt.close()
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
print(len(missing_list), missing_list)

# Memory
airports = None
airports_loc = None
print(df)
# Origin
df['OriginLon'] = df.loc[~df.origin_airport.isin(missing_list)].origin_airport.apply(lambda x : airport_loc[x]['lon'])
df['OriginLat'] = df.loc[~df.origin_airport.isin(missing_list)].origin_airport.apply(lambda x : airport_loc[x]['lat'])

# Destination
df['DestinationLon'] = df.loc[~df.destination_airport.isin(missing_list)].destination_airport.apply(lambda x : airport_loc[x]['lon'])
df['DestinationLat'] = df.loc[~df.destination_airport.isin(missing_list)].destination_airport.apply(lambda x : airport_loc[x]['lat'])
print(df['OriginLon'])
print(df['DestinationLat'])

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

# Create a simple visualization of day/night distribution
plt.figure(figsize=(8, 6))
day_night_counts = df['day_night'].value_counts()
plt.pie(day_night_counts.values, labels=day_night_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Flights by Day/Night Classification')
plt.axis('equal')
plt.savefig('day_night_distribution.png')
plt.close()

useless_variable = None 

####################
######## NIGHT SCORE
#####################

# Loop for all flight
l_night_score = []
l_adjusted_nightscore = []
l_night_scorerev = []
l_night_scorefull =[]

nb = 30
# ~48min with only 30 Points ? (full dataset)

# create empty arrays for now
trajx = np.array([])
trajy = np.array([])





"""print("INDEXING TOP 5%")
sorteddf = df.sort_values(by = ['contrail_CO2_km', 'total_flight_distance_km'], ascending=[True, True])
len(df)

x = df.loc[index, 'contrail_CO2_km']
print("x:",x)

CFcf_sum=x[:int(.05*(len(x)))].sum()
print("length:",int(.05*(len(x))))
print("xslice:",x[:int(.05*(len(x)))])

print("CF:",df.loc[index,'contrail_CO2_km'])"""

print(df.head)
sorteddf = df.sort_values(by = ['contrail_CO2_km', 'total_flight_distance_km'], ascending=[True, True])
print("sorteddf last 10:",sorteddf['contrail_CO2_km'][65050:65015])
print("sorteddf first 10:",sorteddf['contrail_CO2_km'][:10])

n = math.ceil(0.95 * len(sorteddf['contrail_CO2_km']))
print("n",n)
threshold=sorteddf['contrail_CO2_km'].iloc[n]
print("threshold:",threshold)
dfsubset= df[df['contrail_CO2_km'] >= threshold]
print("dfsubset len:",len(dfsubset))
print("fraction of dfsubset:",len(dfsubset)/len(df))


trajx_high, trajy_high = np.array([]), np.array([])
for row in tqdm(dfsubset[['OriginLon', 'OriginLat', 'DestinationLon', 'DestinationLat', 'first_waypoint_time', 'last_waypoint_time']].values, desc='5% highest'):
    geo = great_circle_coords((row[0], row[1]), (row[2], row[3]), nb=nb)
    trajx = np.concatenate((trajx_high, geo[:, 0]))
    trajy = np.concatenate((trajy_high, geo[:, 1]))


"""sorteddf = df.sort_values(by=['contrail_CO2_km', 'total_flight_distance_km'], ascending=[True, True])
n = math.ceil(0.05 * len(sorteddf['contrail_CO2_km']))  # 5% index
threshold = sorteddf['contrail_CO2_km'].iloc[n-1]  # -1 for zero-based index
dfsubsetreverse = df[df['contrail_CO2_km'] <= threshold]

trajx_low, trajy_low = np.array([]), np.array([])
for row in tqdm(dfsubsetreverse[['OriginLon', 'OriginLat', 'DestinationLon', 'DestinationLat', 'first_waypoint_time', 'last_waypoint_time']].values, desc='5% lowest'):
    geo = great_circle_coords((row[0], row[1]), (row[2], row[3]), nb=nb)
    trajx_low = np.concatenate((trajx_low, geo[:, 0]))
    trajy_low = np.concatenate((trajy_low, geo[:, 1]))"""



    

for lona, lata, lonb,latb, time1, time2 in tqdm(df[['OriginLon',
                                                'OriginLat',
                                                'DestinationLon',
                                                'DestinationLat',
                                                'first_waypoint_time',
                                                'last_waypoint_time']].values, desc = 'Night score'):

    # Compute geodesic flight path
    #geo =  great_circle_coords(dep, arr, nb = nb)
    geo =  great_circle_coords((lona, lata), (lonb, latb), nb = nb)
    

    # Time should be a list of times as the plane advance
    times = pd.date_range(time1,
                time2,
                periods = nb)
    # Format
    times = [x.strftime('%Y-%m-%d %H:%M:%S') for x in times]

    # Compute when the sun is up
    #sun_is_up = [sunup(lon, lat, time ) for lon, lat, time in zip(geo[:, 0], geo[:, 1], times)]
    # get angle for how direct the sunlight is
    sun_vals = [sunradians(lon, lat, time ) for lon, lat, time in zip(geo[:, 0], geo[:, 1], times)]
    sun_is_up = [1 if k > 0 else 0 for k in sun_vals]
    sun_valsrev = [sunradians(lon, lat, time ) for lon, lat, time in zip(geo[:, 0], geo[:, 1], times)]
    sun_is_uprev = [1 if k > 0 else 0 for k in sun_vals]
    sun_valsfull = [sunradians(lon, lat, time ) for lon, lat, time in zip(geo[:, 0], geo[:, 1], times)]
    sun_is_upfull = [1 if k > 0 else 0 for k in sun_vals]

    l_night_score.append(100 * (1 - np.sum(sun_is_up) / nb))
    l_adjusted_nightscore.append(100 * (1 - np.sum(sun_vals) / nb))
    l_night_scorerev.append(100 * (1 - np.sum(sun_is_uprev) / nb))
    l_night_scorefull.append(100 * (1 - np.sum(sun_is_upfull) / nb))
# # Add your night score to the dataset
df['night_score'] = l_adjusted_nightscore
df['night_score_bool'] = l_night_score
df['night_score_boolrev'] = l_night_scorerev
df['night_score_boolfull'] = l_night_scorefull

# Note: day_night column was already added earlier in the script

# Compare day/night classification with night score (now that both are available)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
df.boxplot(column='night_score', by='day_night', ax=plt.gca())
plt.title('Night Score by Day/Night Classification')
plt.suptitle('')  # Remove default suptitle

plt.subplot(1, 2, 2)
df.boxplot(column='contrail_energy_forcing', by='day_night', ax=plt.gca())
plt.title('Contrail Energy Forcing by Day/Night Classification')
plt.suptitle('')  # Remove default suptitle

plt.tight_layout()
plt.savefig('day_night_comparison.png')
plt.close()

# # Memory
l_night_score, l_adjusted_nightscore, l_night_scorerev, l_night_scorefull = None, None, None, None


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
print(world['CONTINENT'])
#  We group by similar origin and destination
agg = df.groupby(['Origin_Region',
            'Destination_Region']).agg(
                number=('total_contrail_energy_forcing', 'count'),
            )
print(agg)
# Preparing the plot by making it a table
agg = agg.pivot_table('number', index = 'Origin_Region',
            columns = 'Destination_Region')
print(agg)
from matplotlib.collections import PatchCollection

# vertical
N = agg.index.size
# horizontal
M = agg.columns.size
# Vertical
ylabels = agg.index
# Horizontal
xlabels = agg.columns

x, y = np.meshgrid(np.arange(M), np.arange(N))
s = agg.fillna(0).values
c = agg.fillna(0).values

fig, ax = plt.subplots()

print(s)
print(s.max)

R = s/s.max()/4
circles = [plt.Circle((j,i), radius=np.sqrt(r)) for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap="plasma")
ax.add_collection(col)

ax.set_yticks(np.arange(N), ylabels)
ax.set_xticks(np.arange(M), xlabels, rotation = 60)
ax.set_xticks(np.arange(M+1)-0.5, minor=True,)
ax.set_yticks(np.arange(N+1)-0.5, minor=True)
ax.grid(which='minor')

plt.xlabel('Destination', fontsize = 16)
plt.ylabel('Origin', fontsize = 16)

fig.colorbar(col, label = 'Number of flights')
plt.savefig('#ofFlights.png')

(agg ).style.background_gradient('coolwarm').format(lambda x : round(x, 2)) #* 100 /agg.sum().sum() if we want %



#We define the impact of contrails per km
df['contrail_energy_forcing'] = df['total_contrail_energy_forcing'] / df['total_flight_distance_km']

#  We group by similar origin and destination
agg = df[['contrail_energy_forcing',
          'Origin_Region',
            'Destination_Region']].groupby(['Origin_Region',
            'Destination_Region']).agg(
                         ['mean', 'median',
                      percentile(.25), percentile(.75)]

            )

# Preparing the plot by making it a table
agg = agg.pivot_table('contrail_energy_forcing', index = 'Origin_Region',
            columns = 'Destination_Region')

agg


from matplotlib.collections import PatchCollection

agg_func = 'mean'

# vertical
N = agg.index.size
# horizontal
M = agg[agg_func].columns.size
# Vertical
ylabels = agg.index
# Horizontal
xlabels = agg[agg_func].columns

x, y = np.meshgrid(np.arange(M), np.arange(N))
s = agg[agg_func].fillna(0).values
c = agg[agg_func].fillna(0).values

fig, ax = plt.subplots()

R = s/s.max()/4
circles = [plt.Circle((j,i), radius=np.sqrt(r)) for r, j, i in zip(R.flat, x.flat, y.flat)]
col = PatchCollection(circles, array=c.flatten(), cmap="magma_r")
ax.add_collection(col)

ax.set_yticks(np.arange(N), ylabels)
ax.set_xticks(np.arange(M), xlabels, rotation = 60)
ax.set_xticks(np.arange(M+1)-0.5, minor=True,)
ax.set_yticks(np.arange(N+1)-0.5, minor=True)
ax.grid(which='minor')

plt.xlabel('Destination', fontsize = 16)
plt.ylabel('Origin', fontsize = 16)
plt.title('Aggregation function = ' + agg_func)
fig.colorbar(col, label = agg_func + 'Contrail energy forcing $(J/km)$')
plt.savefig('eforcing.png')
# Simple path geometry to check
l_geo = []

for orlon, orlat, deslon, deslat in tqdm(df[['OriginLon', 'OriginLat','DestinationLon', 'DestinationLat']].values):
  l_geo.append(LineString([(orlon, orlat), (deslon, deslat)]))

# Creation of geodataframe
gdf = gpd.GeoDataFrame(df, geometry = l_geo, crs = 'epsg:4326')
gdf['contrail_impact'] = gdf['total_contrail_energy_forcing'] / gdf['total_flight_distance_km']
gdf.sort_values(by = 'contrail_impact', ascending = False, inplace=True)

# World background
ax = gpd.read_file("ne_110m_admin_0_countries.zip").plot(color = 'white',
                          edgecolor = 'black',
                          lw = .2,
                                                                     )
# Plot flight with a lot of impact 1000 worst
gdf[:1000].plot(ax=ax,
                lw = .1,
                alpha = .3,
               #column = 'contrail_impact'
                )

plt.title('Most critical contrail impact location')

############################
#### Plot

# This classifies the data into night score bins
bounds = np.linspace(0, 100, 11)
distrib = []
for kmin, kmax in zip(bounds[:-1], bounds[1:]):
    #print(kmin, kmax)
    # We save the distribution in each bin
    sub = df[(df.night_score >= kmin) & (df.night_score < kmax)]
    distrib.append(sub.contrail_energy_forcing)
    #Memory
    sub = None
    
# Baseline at 0 impact
plt.hlines(0, 1, 10, ls = '--', lw = 1, color = 'grey')


# We display boxplots in each bin
# we can see 15th and 85th percentile, the quartiles, median (line) and mean (diamond)
plt.boxplot(distrib,
            showfliers = False, # To avoid showing outliers
            whis = (15,85), # To decide the extent of the whis, here 15th and 85th
            showmeans = True,
            # Style
            medianprops = dict(linestyle='solid', color='firebrick', linewidth = 2),
            meanprops = dict(marker='D', markeredgecolor='black',
                    markerfacecolor='firebrick'))
# Memory
distrib = None
    
plt.xticks(range(1, 11), [
    '0-10%',
    '10-20%',
    '20-30%',
    '30-40%',
    '40-50%',
    '50-60%',
    '60-70%',
    '70-80%',
    '80-90%',
    '90-100%'
], fontsize = 8)
plt.xlabel('Night score = $ 1 - \\frac{1}{N} \\sum sin(\\alpha)$')
plt.ylabel('Contrail energy forcing $(J/km)$')

plt.gca().twinx().hist(.5 + df.night_score / 10, bins = 30, alpha = .2, zorder = -np.inf)
plt.yticks([])

plt.savefig('Night_score_contrail_impact.png')
plt.close()
###################################################################

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






h, xedges, yedges, im=plt.hist2d(trajx,trajy,bins=[500,300],cmap=custom_cmap,cmin=1,norm=norm)
# Check that path look ok on a sample (otherwise too big)
#gdf[20000:30000].plot('OriginLon',ax=ax,
#                  lw = .1, kind='kde',)
plt.ylim(-60,80)

# Add colorbar with cutoff indicator
cbar = plt.colorbar(im, extend='max', shrink=0.7)
cbar.set_label('# of flights passing through point')

plt.savefig('Fig1/p2trajmap(datasubset).eps')
plt.savefig('Fig1/p2trajmap(datasubset).jpg')

plt.close()


# This is useful to compute distance
geod = Geod(ellps = 'WGS84')

# Recompute the total distance according to our geometries
gdf['distance_km'] = gdf.geometry.apply(lambda x : geod.geometry_length(x) /1e3)

# # Compare both distances
plt.scatter(
    gdf['total_flight_distance_km'],
    gdf['distance_km'],
    alpha = .1,
    s = 5
)
plt.plot(
    [0, 16000],
    [0, 16000],
    ls = '--',
    c = 'k'
)
plt.xlabel('Given distance km')
plt.ylabel('Computed distance km') 

plt.savefig('Distance_differences.png')
plt.close()



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



######## PLOTTING ########################
# This classifies the data into night score bins
bounds = np.linspace(0, 100, 11)
distrib = []
for kmin, kmax in zip(bounds[:-1], bounds[1:]):
  #print(kmin, kmax)
  # We save the distribution in each bin
  sub = gdf[(gdf.land_score >= kmin) & (gdf.land_score < kmax)]
  distrib.append(sub.contrail_energy_forcing )

sub = None
# Baseline at 0 impact
plt.hlines(0, 1, 10, ls = '--', lw = 1, color = 'grey')


# We display boxplots in each bin
# we can see 15th and 85th percentile, the quartiles, median (line) and mean (diamond)
plt.boxplot(distrib,
            showfliers = False, # To avoid showing outliers
            whis = (15,85), # To decide the extent of the whis, here 15th and 85th
            showmeans = True,
            # Style
            medianprops = dict(linestyle='solid', color='firebrick', linewidth = 2),
            meanprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick'))
distrib = None
plt.xticks(range(1, 11), [
    '0-10%',
    '10-20%',
    '20-30%',
    '30-40%',
    '40-50%',
    '50-60%',
    '60-70%',
    '70-80%',
    '80-90%',
    '90-100%'
], fontsize = 8)
plt.xlabel('Land score (%)')
plt.ylabel('Contrail emissions $(kg_{CO_2}/km)$')

plt.gca().twinx().hist(.5 + gdf.land_score * 10, bins = 30, alpha = .2, zorder = -np.inf)
plt.yticks([])

plt.savefig('land_score_effect.png')
plt.close()


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

############HUB AND SPOKE ANALYSIS###########

"""df['contrail_CO2'] = df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth)
df['contrail_CO2_km'] = df['contrail_CO2'] / df['total_flight_distance_km']
print(df.columns)
#print(df['destination_airport'])
print("######")

count=(df['destination_airport']).value_counts()

hub_airports = []
count_slice_items = count[0:20].items()
for z in count_slice_items:
    hub_airports.append( z[0] )

print ("top airports: ",hub_airports)
hub_airports=[
    "KATL", "KDFW", "KDEN", "KORD", "OMDB",  # 1-5
    "KLAX", "LFPG", "EHAM", "VIDP", "LTFM",  # 6-10
    "EDDF", "RKSI", "KJFK", "WSSS", "VTBS",  # 11-15
    "ZBAA", "ZSPD", "KIAH", "EGLL", "KSEA",  # 16-20
    "KMIA", "LEMD", "ZGGG", "KPHX", "RJTT",  # 21-25
    "LIRF", "WMKK", "ZGSZ", "VHHH", "KCLT",  # 26-30
    "KEWR", "YSSY", "VTBD", "EDDM", "KDTW",  # 31-35
    "MMMX", "VABB", "ZUUU", "KPHL", "KPDX",  # 36-40
    "SBGR", "ZSHC", "ZBHD", "ZJSY", "ZUCK",  # 41-45
    "ZJHK", "ZLXY", "ZSCN", "ZYTX", "ZYTL"   # 46-50
]

Season=[]
for l,therows in df.iterrows():
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
df['Season']=Season
print(df['Season'])
#print(" 10 CF values:",df['contrail_CO2_km'][30000:30010])
CF_mean=[]
latarray=np.array([])
lonarray=np.array([])
airportx  =[]
airporty  =[]
airportc=[]
airportcW =[]
airportcSp=[]
airportcSu=[]
airportcA =[]
airportxW,airportxSp,airportxSu,airportxA=[],[],[],[]
# Create a list to track which airports actually have flights
airports_with_flights = []
for ap in hub_airports:
    print("airport:",ap)
    dfap = df[ df[ 'destination_airport']==ap]
    
    # Skip airports that don't have any flights in the dataset
    if len(dfap) == 0:
        print(f"No flights found for airport {ap}, skipping...")
        continue
    
    # Add this airport to our list of airports with flights
    airports_with_flights.append(ap)
    
   # print (" df = ", dfap )
    mean_contrail = dfap['contrail_CO2_km'].mean()
    print("ave CF of Destap in KG CO2/km:",mean_contrail)

    CF_mean.append(mean_contrail)
    #latarray.append(dfap['DestinationLat'])
    #lonarray.append(dfap['DestinationLon'])
    #latarray = np.concatenate( (latarray, dfap['DestinationLat'].values) )
    #print("latarray:"latarray)
    # take the y value (0), add it to the y list
    #lonarray = np.concatenate( (lonarray, dfap['DestinationLon'].values) )
    #print("lonarray:",lonarray)
    print("current airport:",ap)
    #print("destlat:",dfap['DestinationLat'].iloc[0])
    #print("destlon:",dfap['DestinationLon'].iloc[0])
    airportx.append( dfap['DestinationLon'].iloc[0] )
    airporty.append( dfap['DestinationLat'].iloc[0] )
    airportc.append( mean_contrail)
    dfapW = dfap[dfap['Season']=='Winter']
    dfapSp = dfap[dfap['Season']=='Spring']
    dfapSu = dfap[dfap['Season']=='Summer']
    dfapA = dfap[dfap['Season']=='Autumn']
    print("dfapW len:",len(dfapW['contrail_CO2_km']))
    print("dfapSp len:",len(dfapSp['contrail_CO2_km']))
    print("dfapSu len:",len(dfapSu['contrail_CO2_km']))
    print("dfapA len:",len(dfapA['contrail_CO2_km']))
    print("dfapglobal len:",len(dfap['contrail_CO2_km']))


    
    airportcW.append(dfapW['contrail_CO2_km'].mean())
    airportcSp.append( dfapSp['contrail_CO2_km'].mean())
    airportcSu.append( dfapSu['contrail_CO2_km'].mean())
    airportcA.append( dfapA['contrail_CO2_km'].mean())
    print(dfap['contrail_CO2_km'].mean())
    print(dfapW['contrail_CO2_km'].mean())











print("CF list:",airportc)
print("Airports with flights:", airports_with_flights)
print("Number of airports with flights:", len(airports_with_flights))
fig, ax = plt.subplots()

fig = plt.bar(x=airports_with_flights,height=CF_mean)
#ax.set_xlim(-1,100)
plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')
plt.ylabel("mean CF flights at given hub in KG CO2/km:")
plt.savefig("Fig1/p3bar.eps")
plt.savefig("Fig1/p3bar.jpg")
plt.close()

sznlist=["W","Sp",'Su',"A","global"] 
sznsCF=[airportcW,airportcSp,airportcSu,airportcA,airportc]
for i,cf in enumerate(sznsCF):
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

    # Create normalization with custom range
    norm = colors.Normalize(vmin=0, vmax=12.5)

    # Create scatter plot with custom colormap
    scatter = plt.scatter(
    airportx, airporty,
    marker='o',
    c=cf,
    cmap=custom_cmap,
    norm=norm
    )
    # Add colorbar with cutoff indicator    
    cbar = plt.colorbar(scatter, extend='max')
    cbar.set_label('CF in (KG CO₂)/Km per Flight')

    plt.xlim(-175,175)
    plt.ylim(-60,80)
    plt.savefig("TestFig1/p3"+sznlist[i]+".eps",dpi=300)
    #plt.savefig("Fig1/p3map.jpg",dpi=300)
    plt.close()

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

# Create normalization with custom range
norm = colors.Normalize(vmin=0, vmax=12.5)

# Create scatter plot with custom colormap
scatter = plt.scatter(
airportx, airporty,
marker='o',
c=airportc,
cmap=custom_cmap,
norm=norm
)
# Add colorbar with cutoff indicator    
cbar = plt.colorbar(scatter, extend='max')
cbar.set_label('CF in (KG CO₂)/Km per Flight')


####AI####### Add labels to each point
for i, (x, y, ap) in enumerate(zip(airportx, airporty, airports_with_flights)):
    # Default offset for most airports
    x_offset = 2
    y_offset = 2
    
    # Special adjustment for JFK
    if ap == 'KJFK':
        y_offset = 10  # Move JFK label upwards
    
    plt.annotate(
        ap,
        (x, y),
        xytext=(x_offset, y_offset),
        textcoords='offset points',
        fontsize=8,
        ha='left',
        va='bottom'
    )
#######
plt.xlim(-150,-50)
plt.ylim(15,50)
plt.savefig('Fig1/p3USmap.eps',dpi=300)
#plt.savefig("Fig1/p3USmap.jpg",dpi=300)
plt.xlim(-10,30)
plt.ylim(45,55)
plt.savefig('Fig1/p3EUmap.eps',dpi=300)
#plt.savefig("Fig1/p3EUmap.jpg",dpi=300)
plt.xlim(49,152)
plt.ylim(-40,50)
plt.savefig('Fig1/p3Asiamap.eps',dpi=300)
#plt.savefig("Fig1/p3Asiamap.jpg",dpi=300)
plt.close()"""


################# NEW METHOD FOR ML NIGHTSCORE #########################



# Decide the different shifts we want to have for teh night score (in hours)

list_shifts = [0, 1, 2, 3, 4, 5, 6] 



from datetime import datetime



def calcnightscore(array_element, nb=30, list_shifts = list_shifts):

    lona, lata, lonb, latb, time1, time2 = array_element

    # Compute geodesic flight path

    geo =  great_circle_coords((lona, lata), (lonb, latb), nb = nb)



    to_return = list()

    to_returnb = list()

    for shift in list_shifts:

      # We shift the time

      time1_shift = time1 + timedelta(hours = shift)

      time2_shift = time2 + timedelta(hours = shift)
     
      # Time should be a list of times as the plane advance

      times = pd.date_range(time1_shift,

                  time2_shift,

                  periods = nb)

      # Format

      times = [x.strftime('%Y-%m-%d %H:%M:%S') for x in times]



      # Compute when the sun is up

      # Full values with average of the sinus

      sun_vals_full = [sunradians(lon, lat, time ) for lon, lat, time in zip(geo[:, 0], geo[:, 1], times)]

      # Boolean indicator

      boolean = [1 if k > 0 else 0 for k in sun_vals_full] 

      # # Adjusted only >0

      # adj_pos = [max(0, x) for x in sun_vals_full]

      # # Adjusted < 0

      # adj_neg =  [min(0, x) for x in sun_vals_full]

      

      to_return.append(100 * (np.mean(sun_vals_full)))

      to_returnb.append(100 * (1- np.mean(boolean)))



    return to_return, to_returnb





####################

######## NIGHT SCORES

#####################





  

##########################################

# Compute on the entire dataframe



# Number of points to interpolate sun score

nb = 30

    

array = list(gdf[['OriginLon',

            'OriginLat',

            'DestinationLon',

            'DestinationLat',

            'first_waypoint_time',

            'last_waypoint_time']].values)

  

res = []

# Parallel iteration

s = time.time()



nightscores_full = []

nightscores_bool = []

calc_dist = []



for l in list_shifts:

   nightscores_full.append( [] ) # make an empty list for each shift

   nightscores_bool.append( [] ) # make an empty list for each shift



geod = Geod(ellps='WGS84')





for flight in array:



   _,_,dist = geod.inv(flight[0],flight[1],flight[2],flight[3])

   calc_dist.append(dist/1000.)



   # calculate the full and bool nightscores from flight data

   resf,resb = calcnightscore(flight)

   print ("flight =", flight, 'res full = ',resf, ' res boolean = ',resb)



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


gdf.to_parquet('testoutputfeatures.pq')



#gdf.to_parquet('mydir/features_'+filter+ '_gdf.pq')
#print(df["first_waypoint_time"].dt.year)

# 7Go --> Too heavy

# --- Night Score Predictiveness Plot ---
"""import matplotlib.pyplot as plt
import numpy as np

# Bin the night score
bins = np.linspace(0, 100, 11)  # 0-10, 10-20, ..., 90-100
if 'night_score' in df.columns:
    df['night_bin'] = pd.cut(df['night_score'], bins, include_lowest=True, right=False)

    # Boxplot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='contrail_energy_forcing', by='night_bin', grid=False)
    plt.title('Contrail Forcing vs. Night Score')
    plt.suptitle('')
    plt.xlabel('Night Score Bin (%)')
    plt.ylabel('Contrail Energy Forcing (J/km)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('nightscore_vs_contrailforcing.png')
    plt.close()

    # Correlation
    corr = df['night_score'].corr(df['contrail_energy_forcing'])
    print(f"Pearson correlation between night score and contrail energy forcing: {corr:.3f}")
# --- End Night Score Predictiveness Plot ---"""


