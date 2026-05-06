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
df = pd.read_parquet("/Users/silas/Documents/Watershed/Xavier_export_contrails/20190101-summary.pq")


print(df["first_waypoint_time"].dt.year)





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

###################################################################

#World background
ax = gpd.read_file("ne_110m_admin_0_countries.zip").plot(color = 'white',
                                                                      edgecolor = 'black',
                                                                      lw = .5)



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

############HUB AND SPOKE ANALYSIS###########

print(df.columns)
print(df['destination_airport'])
print("######")

count=(df['destination_airport']).value_counts()
print("airport counts :",count.head)
count_slice = count[0:20].items()
print("count slice: ", count_slice)

hubSeries= []

#if df['destination_airport']==hubSeries:


#############################################
############## SAVING #######################


gdf.to_parquet('1-1-2019featuresgdf.pq')

# 7Go --> Too heavy


