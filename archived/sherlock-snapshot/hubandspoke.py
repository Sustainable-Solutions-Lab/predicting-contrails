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
df = pd.read_parquet("/Users/silas/Documents/Watershed/Xavier_export_contrails/")#20190101-summary.pq")
df=df[:10000]

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
#print(len(missing_list), missing_list)

# Memory
airports = None
airports_loc = None
#print(df)
# Origin
df['OriginLon'] = df.loc[~df.origin_airport.isin(missing_list)].origin_airport.apply(lambda x : airport_loc[x]['lon'])
df['OriginLat'] = df.loc[~df.origin_airport.isin(missing_list)].origin_airport.apply(lambda x : airport_loc[x]['lat'])

# Destination
df['DestinationLon'] = df.loc[~df.destination_airport.isin(missing_list)].destination_airport.apply(lambda x : airport_loc[x]['lon'])
df['DestinationLat'] = df.loc[~df.destination_airport.isin(missing_list)].destination_airport.apply(lambda x : airport_loc[x]['lat'])
#print(df['OriginLon'])
#print(df['DestinationLat'])

# Drop missing airports
df.dropna(subset = ['OriginLon', 'DestinationLon','OriginLat', 'DestinationLat'], inplace=True)
print("shape after airport clean", df.shape)
useless_variable = None 
############HUB AND SPOKE ANALYSIS###########

df['contrail_CO2'] = df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth)
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
        Season.append("Autumn")
df['Season']=Season
print(df['Season'])
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
    print("destlat:",dfap['DestinationLat'].iloc[0])
    print("destlon:",dfap['DestinationLon'].iloc[0])
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
    #norm = colors.LogNorm()
    # Create scatter plot with custom colormap
    scatter = plt.scatter(
    airportx, airporty,
    marker='o',
    c=cf,
    cmap=custom_cmap,
    norm=norm
    )
    # Add colorbar with cutoff indicator    
    cbar = plt.colorbar(scatter)#, extend='max')
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
#norm = colors.LogNorm()


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
plt.close()




