import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import xarray as xr
import airportsdata
import sys

#############################
# HARD-CODED AIRCRAFT CRUISE ALTITUDES
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
    # Turboprops (examples)
    'AT72': 25000, 'AT75': 25000, 'DH8A': 25000, 'DH8D': 25000,
}
default_altitude = 37000  # If type unknown

# Usual ERA5 pressure levels, in hPa. Used for nearest-match conversion.
ERA5_LEVELS = [
    1000, 925, 900, 850, 800, 700, 600, 500,
    400, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10
]

# Conversion: US Std Atmosphere (simplified barometric formula for 0-11 km)
def feet_to_pressure_level(feet, ERA5_LEVELS):
    meters = feet * 0.3048
    P0 = 1013.25
    T0 = 288.15
    L = 0.0065
    g = 9.80665
    M = 0.0289644
    R = 8.3144598
    # Hypsometric equation:
    P = P0 * (1 - (L * meters) / T0) ** (g * M / (R * L))
    # Find nearest
    arr = np.array(ERA5_LEVELS)
    return arr[np.argmin(np.abs(arr - P))]

def estimate_cruise_pressure(row):
    ac_type = row.get('aircraft_type', '')
    flight_distance = row.get('flight_distance', 1000)
    base_altitude = altitude_estimates.get(ac_type, default_altitude)
    if flight_distance < 300:
        factor = 0.75
    elif flight_distance < 800:
        factor = 0.9
    elif flight_distance < 2000:
        factor = 1.0
    else:
        factor = 1.05
    alt_ft = base_altitude * factor
    return feet_to_pressure_level(alt_ft, ERA5_LEVELS)

def get_airport_coords(df):
    airports = airportsdata.load()
    airport_names = list(airports.keys())
    def lookup(apt, key):
        if apt in airport_names:
            return airports[apt][key]
        return np.nan
    df['OriginLon'] = df['origin_airport'].apply(lambda x: lookup(x, 'lon'))
    df['OriginLat'] = df['origin_airport'].apply(lambda x: lookup(x, 'lat'))
    df['DestLon']   = df['destination_airport'].apply(lambda x: lookup(x, 'lon'))
    df['DestLat']   = df['destination_airport'].apply(lambda x: lookup(x, 'lat'))
    return df

def calcweather(row, ds):
    pressure = estimate_cruise_pressure(row)
    # Best-effort for time parsing (adjust your time col if needed!)
    if 'date' in row:
        date = pd.to_datetime(row['date'])
    elif 'first_waypoint_time' in row:
        date = pd.to_datetime(row['first_waypoint_time'])
    else:
        raise ValueError("No datetime column found!")
    lat = np.nanmean([row['OriginLat'], row['DestLat']])
    lon = np.nanmean([row['OriginLon'], row['DestLon']])
    try:
        subset = ds.sel(
            valid_time=date,
            pressure_level=pressure,
            latitude=lat,
            longitude=lon,
            method='nearest'
        )
       
       # geopot = subset['z'].item()
        humid = subset['r'].item()

    except Exception as e:
        print(f"Weather extraction failed at {date}, {lat}, {lon}, {pressure}hPa: {e}")
        #temp = humid = uwind = vwind = geopot = np.nan
    return  humid

#############################
# MAIN SCRIPT
if len(sys.argv) < 2:
    print("Usage: python add_weather_features_only.py <filter>")
    sys.exit(1)

filter = sys.argv[1]
data_dir = Path('Xavier_export_contrails')

for parquet_file in data_dir.glob(filter + "*-summary.pq"):
    print ("Reading file: "+parquet_file.name)
    df = pd.concat(
        (pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob(filter+"*-summary.pq")),ignore_index=True) 
    print("shape:",df.shape)
    # Remove flights with missing airports
    df = df[
        (~df["destination_airport"].isna())
        & (df["destination_airport"] != "")
        & (~df["origin_airport"].isna())
        & (df["origin_airport"] != "")
    ]
    #df = df.sample(n=1000, random_state=42)
    # Ensure coordinates present
    if not all(x in df.columns for x in ['OriginLon','OriginLat','DestLon','DestLat']):
        print("Adding airport coordinates...")
        df = get_airport_coords(df)

    #era5_file = f"era5_pl_{filter[:-1]}.nc"
    era5_file = 'humidity2019071.nc'
    print("Loading ERA5 data:", era5_file)
    ds = xr.open_dataset(era5_file, engine="netcdf4")

    print("Creating weather columns (humidity)...")
    humids=[]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        h = calcweather(row, ds)
        humids.append(h)
    df['humidity'] = humids

    columns_to_save = ['flight_id', 'humidity']

    df[columns_to_save].to_parquet('atmosonlyfeatures_'+filter+ '_gdf.pq')