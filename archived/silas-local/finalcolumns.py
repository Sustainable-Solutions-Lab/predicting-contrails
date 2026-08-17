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
import requests
from geopy.geocoders import Nominatim
import countryinfo
from countryinfo import CountryInfo
import matplotlib.path as mpltPath
import continent

clean=pd.read_parquet('cleanaddon.pq')

maxval=clean['normcontrailforcing'].max()
minval=clean['normcontrailforcing'].min()
clean['ncf100']=(clean["normcontrailforcing"]-minval)/(maxval-minval)

print(clean["first_waypoint_time"])
clean['contrailimpact'] = [1 if x>0.00001 else -1 if x<-0.00001 else 0 for x in clean['normcontrailforcing']]
clean['overland'] = [1 if x==100 else 0 for x in clean['transoceanic']]
flightlength=[]
for i,row in clean.iterrows():
	if row['total_flight_distance_km']<=1000:
		flightlength.append("short")
	if row['total_flight_distance_km']>1000:
		flightlength.append("long")
clean['shortflight']=flightlength

print(clean['Season'])
print(clean['shortflight'])
c = continent.continentdb()

continentlist=[]
for i,row in clean.iterrows():
	res = c.get_cont(row['OriginLat'],row['OriginLon'])

	continentlist.append(res)
clean['continent']=continentlist
clean.to_parquet('cleanwcont.pq')