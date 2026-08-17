import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import math
import geopy.distance
from math import sin, cos, atan2, sqrt, radians
from pyproj import Geod
from global_land_mask import globe
import pyarrow
import allairports

allflightsdf = pd.read_parquet('dataflights/')
print(f"{len(allflightsdf)} flights")



allflightsdf.columns=['flight_id', 'callsign', 'icao_address', 'flight_number', 'tail_number',
       'aircraft_type_icao', 'aircraft_engine_type', 'origin_airport',
       'origin_airport_name', 'origin_country', 'destination_airport',
       'destination_airport_name', 'destination_country',
       'first_waypoint_time', 'last_waypoint_time', 'flight_duration_h',
       'total_flight_distance_km', 'n_wypts', 'load_factor', 'total_fuel_burn',
       'engine_name', 'engine_uid', 'mean_aircraft_mass',
       'mean_sdr_flight_waypoints', 'mean_olr_flight_waypoints',
       'mean_contrail_lifetime', 'max_contrail_lifetime',
       'total_contrail_length_sac_km', 'total_persistent_contrail_length_km',
       'total_contrail_energy_forcing']

clean = allflightsdf.loc[
    # remove fields with null descriptors
    ~allflightsdf["callsign"].isnull() & \
    ~allflightsdf["origin_airport"].isnull() & \
    ~(allflightsdf["origin_airport"] == "") & \
    ~allflightsdf["destination_airport"].isnull() & \
    ~(allflightsdf["destination_airport"] == "") & \
    ~allflightsdf["tail_number"].isnull() & \
    ~allflightsdf["aircraft_type_icao"].isnull() & \
    ~allflightsdf["total_contrail_energy_forcing"].isnull() & \
    # general aviation
    ~(allflightsdf["tail_number"].str.startswith("G-")).astype(bool)
]
print(f"Removed {len(allflightsdf) - len(clean)} flights")

print(clean['total_contrail_energy_forcing'])
print(clean['origin_airport'])

airports = allairports.load()  # key is the ICAO identifier (the default)

dict_keys=airports.keys()
lonlist=[]
latlist=[]
destlonlist=[]
destlatlist=[]
tally=0
missingcode=[]
for i,row in clean.iterrows():
	if row['origin_airport'] in airports.keys():
		lon0=airports[row['origin_airport']]['lon']
		lat0=airports[row['origin_airport']]['lat']
		lonlist.append(lon0)
		latlist.append(lat0)
	else:
		print(row['origin_airport'], "missing")
		lonlist.append(0)
		latlist.append(0)
		tally=tally+1
		if not row['origin_airport'] in missingcode:
			missingcode.append(row['origin_airport'])
	if row['destination_airport'] in airports.keys():
		lon1=airports[row['destination_airport']]['lon']
		lat1=airports[row['destination_airport']]['lat']
		destlonlist.append(lon1)
		destlatlist.append(lat1)
	else:
		print(row['destination_airport'], "missing")
		destlonlist.append(0)
		destlatlist.append(0)
		tally=tally+1
		if not row['destination_airport'] in missingcode:
			missingcode.append(row['destination_airport'])
print(missingcode)
clean['DestinationLon']=destlonlist
clean['OriginLat']=latlist
clean['OriginLon']=lonlist
clean['DestinationLat']=destlatlist

#copied from colab
print(clean['total_contrail_energy_forcing'])
AREA_EARTH = 5.101e14
SECONDS_PER_YEAR = 60 * 60 * 24 * 365
AGWP100 = 92.5 * 1e-15 * SECONDS_PER_YEAR
print(AGWP100)
conversion= (AGWP100 * AREA_EARTH) / 1e3 
clean['CO2_tons']=clean['total_contrail_energy_forcing']/ conversion
print(clean['CO2_tons'])

def strtoblock(datestr):
# print(datestr)
  datestr=datestr.split()
  timelist=datestr[1].split(":")
  blocks=int(float(timelist[0])/4.)#+int(timelist[1])/60.0
  return blocks

def strtohour(datestr):
# print(datestr)
  datestr=datestr.split()
  timelist=datestr[1].split(":")
  minutes=int(float(timelist[0]))+int(timelist[1])/60.0
  return minutes
#?
#clean['last_waypoint_time'].dropna(how="all", inplace=True)


blocktomidnight=[]
MinMidnight=[]
for i in clean["first_waypoint_time"]:
  Newval=i.hour + (i.minute)/60.0
  Newval2=(i.hour)/4
  MinMidnight.append(Newval)
  blocktomidnight.append(Newval2)

clean['Hours_Since_Midnight']=MinMidnight
clean['blocks_to_midnight']=blocktomidnight
#print(allflightsdf['blocks_to_midnight'])
#rint(allflightsdf)

#print(allflightsdf.first_waypoint_time.describe)
#sns.violinplot(data=allflightsdf, x="first_waypoint_time", y="NewMethod")
#plot=sns.violinplot(x=allflightsdf["TripID"])
#fig=plot.get_figure()
#fig.savefig("TripID.png")
#print(allflightsdf["TripID"].head(3))
def calc_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)
    return brng
def calc_dist(coords_1, coords_2):
    dist = geopy.distance.geodesic(coords_1, coords_2).km
    return dist

def genoceanic(lon0, lat0, lon1, lat1, n_extra_points):
    geoid = Geod(ellps="WGS84")
    extra_points = geoid.npts(lon0, lat0, lon1, lat1, n_extra_points)
    return extra_points

 
def perc_transoceanic(lon0, lat0, lon1, lat1, pointcount):

    extra_points = genoceanic(lon0, lat0, lon1, lat1, pointcount)
    tally=0
    for i in range(len(extra_points)):
        pair = extra_points[i]
        pairlon=pair[0]
        pairlat=pair[1]
        ans=globe.is_land(pairlat,pairlon)
        if ans==True:
            tally=tally+1
    perctrue=tally*100.0/len(extra_points)
    return perctrue



Bearingcol=[]
Directioncol=[]
#use range length for loop and index it in all four columns
#for latia in subset['OriginLat']:
for j,row in clean.iterrows():
  latia=row['OriginLat']
  latib=row['DestinationLat']
  longa=row['OriginLon']
  longb=row['DestinationLon']
  bearing=calc_bearing(latia,latib,longa,longb)
  if bearing < 0:
    bearing=360+bearing
  Bearingcol.append(bearing)
  direction=round(bearing/45.00)
  if direction==8:
    direction=0
  Directioncol.append(direction)

clean['Direction']=Directioncol
clean["Bearing"]=Bearingcol

print(clean['total_flight_distance_km'])
print(clean['flight_duration_h'])

Speed=[]
for lt,row1 in clean.iterrows():
  flspeed=float(row1['total_flight_distance_km'])/float(row1['flight_duration_h'])
  Speed.append(flspeed)
clean['Speed']=Speed

speedbearing=[]
for jk,row2 in clean.iterrows():

  bearingmultiplier=math.sin(row2['Bearing']*(math.pi/180.0))

  spbr=row2['Speed']*bearingmultiplier
  speedbearing.append(spbr)
clean['speedbearing']=speedbearing



Transoceanic = []
for r,therow in clean.iterrows():
	pointnum=round(therow['total_flight_distance_km']/150.00)
	pointnum=max(5, pointnum)
	ans = perc_transoceanic(therow['OriginLon'],therow['OriginLat'],therow['DestinationLon'],therow['DestinationLat'],pointnum)
	Transoceanic.append(ans)
clean['transoceanic']=Transoceanic
print(clean['transoceanic'].head(100))
dawndusk= []
for l,row in clean.iterrows():
  if 4 < row['Hours_Since_Midnight'] < 7:
    dawndusk.append('dawn')
  else: 
    if 16 < row['Hours_Since_Midnight'] < 20:
      dawndusk.append('dusk')
    else:
      dawndusk.append('blank')
clean["dawndusk"]=dawndusk
print(clean["dawndusk"])

dawndusk= []
for l,row in clean.iterrows():
  if 4 < row['Hours_Since_Midnight'] < 7 and row['speedbearing'] > 400:
    dawndusk.append('dawn')
  else: 
    if 16 < row['Hours_Since_Midnight'] < 20 and row['speedbearing'] < -400:
      dawndusk.append('dusk')
    else:
      dawndusk.append('blank')

clean["dawndusk"]=dawndusk
print(clean["dawndusk"])

normalizedcontrailforcing=[]
for l,rows in clean.iterrows():
  normalize=float(rows['CO2_tons'])/rows['total_flight_distance_km']
  normalizedcontrailforcing.append(normalize)
clean['normcontrailforcing']=normalizedcontrailforcing
Season=[]
for l,therows in clean.iterrows():
	month= therows['first_waypoint_time'].month
	if month in {12,1,2}:
		Season.append('Winter')
	if month in {3,4,5}:
		Season.append("Spring")
	if month in {6,7,8}:
		Season.append("Summer")
	if month in {9,10,11}:
		Season.append("Fall")
clean['Season']=Season
print(clean['Season'])
Dayscol=[]
for l,therows in clean.iterrows():
	days= therows['first_waypoint_time'].day
	Dayscol.append(days)
clean['Days']=Dayscol
print(clean['Days'])

clean.to_parquet('cleanaddon.pq')