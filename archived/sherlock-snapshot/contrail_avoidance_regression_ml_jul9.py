##################
### Librairies ###
##################

#Warning
import warnings
warnings.filterwarnings('ignore')

#Loggings
import logging
import sys

logging.basicConfig(filename='./results/contrail_avoidance_regression_ml.txt', encoding='utf-8',
 level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p') #%m/%d/%Y - pour la date

# Classics
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
from collections import Counter
import math

# Others
from tqdm import tqdm

# Machine Learning
# Models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
# Metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
def root_mean_squared_error(y, yp) :
    return np.sqrt(mean_squared_error(y, yp))
    
# Methods
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
# Imbalance learning
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

####################
### Load Dataset ###
####################

path = '../../../../../surface3/xbonnema/contrail_avoidance/'

# Read new
print("Loading dataset...")
import glob

"""# Find all 2019 files
file_pattern = "/Users/silas/Documents/Watershed/Sherlockdfs/features_2019*_gdf.pq"
files_2019 = glob.glob(file_pattern)
print(f"Found {len(files_2019)} 2019 files: {files_2019}")

# Read and concatenate all 2019 files
dfs = []
for file in files_2019:
    print(f"Loading {file}...")
    df_temp = pd.read_parquet(file)
    dfs.append(df_temp)
    print(f"  - Loaded {df_temp.shape[0]:,} flights")

# Concatenate all dataframes
df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset: {df.shape[0]:,} flights from {len(files_2019)} files")
"""

#df = pd.read_parquet("/Users/silas/Documents/Watershed/Sherlockdfs/features_2019043_gdf.pq")
df= pd.read_parquet("/home/groups/sjdavis/silas/Watershed/Sherlockdfs/features_2019043_gdf.pq")
msg = 'Number of flights: '+str(df.shape[0])
logging.info(msg)
print(f"Loaded {df.shape[0]:,} flights")

msg = 'Size of dataset (Mo): ' + str(sys.getsizeof(df) >> 20)
logging.info(msg)
print(f"Dataset size: {sys.getsizeof(df) >> 20} MB")

# Check for night_score_bool_1 values before filtering
print(f"Before distance filter - Unique values in night_score_bool_1: {df['night_score_bool_1'].unique()}")
print(f"Before distance filter - Count of values == 100: {(df['night_score_bool_1'] == 100).sum()}")
print(f"Before distance filter - Count of values != 100: {(df['night_score_bool_1'] != 100).sum()}")

# Remove flights with very short or zero distance to avoid inf in contrail_CO2_km
min_distance = 0.1  # km
distance_filter = (df['total_flight_distance_km'].astype(float) >= min_distance) & (df['total_flight_distance_km'].astype(float) > 0)
print(f"Distance filter: {np.sum(~distance_filter)} flights removed (distance < {min_distance} km or = 0)")
df = df[distance_filter]

# Verify the filter worked
print(f"After filtering - Min distance: {df['total_flight_distance_km'].min()}")
print(f"After filtering - Max distance: {df['total_flight_distance_km'].max()}")
print(f"After filtering - Zero distances: {(df['total_flight_distance_km'] == 0).sum()}")

# Check for night_score_bool_1 values after filtering
print(f"After distance filter - Unique values in night_score_bool_1: {df['night_score_bool_1'].unique()}")
print(f"After distance filter - Count of values == 100: {(df['night_score_bool_1'] == 100).sum()}")
print(f"After distance filter - Count of values != 100: {(df['night_score_bool_1'] != 100).sum()}")

# Check variance in night_score_full_0
print(f"Night score full_0 variance: {df['night_score_full_0'].var()}")
print(f"Night score full_0 unique values: {df['night_score_full_0'].unique()}")
print(f"Night score full_0 min: {df['night_score_full_0'].min()}")
print(f"Night score full_0 max: {df['night_score_full_0'].max()}")
print(f"Night score full_0 mean: {df['night_score_full_0'].mean()}")
print(f"Night score full_0 std: {df['night_score_full_0'].std()}")

AGWP_100 = 82.5 * 10**(-15) # W/m²/kg
EI_CO2 = 3.16 # kgCO2/kg fuel
S_earth = 5.101 * 10**14 # m²
df['contrail_CO2'] = df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth)
df['contrail_EF_km'] = df['total_contrail_energy_forcing'] / df['total_flight_distance_km']
df['contrail_CO2_km'] = df['contrail_CO2'] / df['total_flight_distance_km']

#######################
### Encoding variables
######################

encoding = True

if encoding :
    print("Creating features...")
    
    # Encoding time
    df['first_waypoint_time'] = pd.to_datetime(df['first_waypoint_time'])
    
    # target
    df['contrail_CO2'] = df['total_contrail_energy_forcing'] /  (AGWP_100 * 365 * 24 * 60 * 60 * S_earth)
    df['contrail_CO2_km'] = df['contrail_CO2'] / df['total_flight_distance_km']

    print("Creating time features...")
    # Time
    df['day_sin'] = df.first_waypoint_time.apply(lambda x : math.sin(np.pi * x.dayofyear * 2 / 365))
    df['day_cos'] = df.first_waypoint_time.apply(lambda x : math.cos(np.pi * x.dayofyear * 2 / 365))
    
    # Start time features
    df['start_hour_sin'] = df.first_waypoint_time.apply(lambda x : math.sin(np.pi * x.hour * 2 / 24))
    df['start_hour_cos'] = df.first_waypoint_time.apply(lambda x : math.cos(np.pi * x.hour * 2 / 24))
    
    # End time features
    df['end_hour_sin'] = df.last_waypoint_time.apply(lambda x : math.sin(np.pi * x.hour * 2 / 24))
    df['end_hour_cos'] = df.last_waypoint_time.apply(lambda x : math.cos(np.pi * x.hour * 2 / 24))
    
    # Middle of flight time
    df['mid_flight_time'] = df['first_waypoint_time'] + (df['last_waypoint_time'] - df['first_waypoint_time']) / 2
    df['mid_hour_sin'] = df.mid_flight_time.apply(lambda x : math.sin(np.pi * x.hour * 2 / 24))
    df['mid_hour_cos'] = df.mid_flight_time.apply(lambda x : math.cos(np.pi * x.hour * 2 / 24))
    
    # Time span features
    df['time_span_hours'] = (df['last_waypoint_time'] - df['first_waypoint_time']).dt.total_seconds() / 3600
    df['crosses_midnight'] = (df['last_waypoint_time'].dt.date != df['first_waypoint_time'].dt.date).astype(int)
    print("Sample of crosses_midnight column:")
    print(df[['first_waypoint_time', 'last_waypoint_time', 'crosses_midnight']].head(10))
    
    print("Creating geographic features...")
    # Bearing
    df['bearing_sin'] = df.bearing.apply(lambda x : math.sin(np.pi * x * 2 / 360))
    df['bearing_cos'] = df.bearing.apply(lambda x : math.cos(np.pi * x * 2 / 360))
    
    #Longitude
    df['OriginLon_sin'] = df.OriginLon.apply(lambda x : math.sin(np.pi * x * 2 / 360))
    df['OriginLon_cos'] = df.OriginLon.apply(lambda x : math.cos(np.pi * x * 2 / 360))
    df['DestinationLon_sin'] = df.DestinationLon.apply(lambda x : math.sin(np.pi * x * 2 / 360))
    df['DestinationLon_cos'] = df.DestinationLon.apply(lambda x : math.cos(np.pi * x * 2 / 360))

    print("Concatenating features...")
    # Dummies
    df = pd.concat([
            df[[
                'contrail_CO2_km',
                'total_flight_distance_km',
                'day_sin', 'day_cos',
                'start_hour_sin', 'start_hour_cos',
                'end_hour_sin', 'end_hour_cos',
                'mid_hour_sin', 'mid_hour_cos',
                'time_span_hours', 'crosses_midnight',
                'land_score', 
                'bearing_sin','bearing_cos',
                'OriginLon_sin','DestinationLon_sin',
                'OriginLon_cos','DestinationLon_cos',
                'OriginLat','DestinationLat'
        ]],
            df[
    ['night_score_full_'+str(shift) for shift in [0, 1, 2, 3]]
                ],
            df[
    ['night_score_bool_'+str(shift) for shift in [0, 1, 2, 3]]
                ]
        ], 
                axis = 1)

# Threshold definition - 10% worst flights
th = df.contrail_CO2_km.quantile(.95)

# Encoding 'regression - Fix to prevent NaN and inf values
# Handles edge cases like zeros and negative values
df['contrail_type'] = df['contrail_CO2_km'].copy()

# For zero values, set to 0
df.loc[df['contrail_CO2_km'] == 0, 'contrail_type'] = 0

# For negative values, use negative log of absolute value
negative_mask = df['contrail_CO2_km'] < 0
df.loc[negative_mask, 'contrail_type'] = -np.log(np.abs(df.loc[negative_mask, 'contrail_CO2_km']))

# For positive values, use positive log
positive_mask = df['contrail_CO2_km'] > 0
df.loc[positive_mask, 'contrail_type'] = np.log(df.loc[positive_mask, 'contrail_CO2_km'])

# Debug: Check for any remaining inf or NaN values
inf_count = np.isinf(df['contrail_type']).sum()
nan_count = df['contrail_type'].isna().sum()
print(f"After contrail_type calculation - inf values: {inf_count}")
print(f"After contrail_type calculation - NaN values: {nan_count}")

if inf_count > 0 or nan_count > 0:
    print("WARNING: Still have inf or NaN values in contrail_type!")
    print(f"Sample contrail_CO2_km values causing issues:")
    problem_mask = np.isinf(df['contrail_type']) | df['contrail_type'].isna()
    print(df.loc[problem_mask, 'contrail_CO2_km'].head(10))

msg = 'df after features = ' + str(df.head())
logging.info(msg)

msg = 'Size of dataset after feature selection (Mo): ' + str(sys.getsizeof(df) >> 20)

logging.info(msg)

# Types
df = df.astype({
    'day_sin': 'float16',
    'day_cos': 'float16',
    'contrail_CO2_km': 'float64',
    'contrail_type': 'float64',
    'total_flight_distance_km': 'float64',
    'land_score': 'float16',
    'bearing_sin': 'float16',
    'bearing_cos': 'float16',
    'OriginLon_sin' : 'float16',
    'DestinationLon_sin' : 'float16',
    'OriginLon_cos' : 'float16',
    'DestinationLon_cos' : 'float16',
    'OriginLat': 'float16',
    'DestinationLat': 'float16',
})

logging.info("Checking for NaNs")
for col in df.columns:
    numnan = df[col].isnull().sum()
    msg = 'NaNs in column ' + col + ' = ' + str(numnan)
    logging.info(msg)
logging.info("Done checking for NaNs")

msg = 'df after dropping = ' + str(df.head())
logging.info(msg)

logging.info("Checking for NaNs 2nd time")
for col in df.columns:
    numnan = df[col].isnull().sum()
    msg = 'NaNs in column ' + col + ' = ' + str(numnan)
    logging.info(msg)
logging.info("Done checking for NaNs")

logging.info("Checking for inf")
for col in df.columns:
    numinf = np.isinf(df[col]).sum()
    msg = 'inf in column ' + col + ' = ' + str(numinf)
    logging.info(msg)
logging.info("Done checking for infs")

msg = 'Size of dataset after type conversion (Mo): ' + str(sys.getsizeof(df) >> 20)
logging.info(msg)

msg = str(df.columns)
logging.info(msg)

# Features and variables
target_variable = 'contrail_type'

# Features
X = df.drop([target_variable, 'contrail_CO2_km', 'total_flight_distance_km'], axis = 1)
# Labels
y = df[target_variable]

# for later: impact in terms of radiative effect per km - could be absolute too
impact = df[['contrail_CO2_km', 'total_flight_distance_km']]

# save some RAM
df = None

########################
### Train/Test Split ###
########################

# We define training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state = 42,
    test_size = 0.3, # How many % are attributed to test ? 
    )

# X_test and y_test are only use for final validation

# Memory
X, y = None, None

######################
### Grid Search CV ###
######################

# Model
model = XGBRegressor(
                    n_jobs = -1,
                    learning_rate=.5, 
)

# Wich paramaters do we want to try
parameters = {
             'n_estimators' : [10, 10000],
              'max_depth' : [5, 7],
              }

# Calculate total iterations for progress bar
total_iterations = len(parameters['n_estimators']) * len(parameters['max_depth']) * 3  # 3-fold CV
print(f"GridSearchCV will run {total_iterations} total iterations")

# Define the scores and the cv
clf = GridSearchCV(model, parameters, 
                   cv = 3, 
                   scoring = {'RMSE':'neg_root_mean_squared_error',
                              'R2':'r2', 
                              'MAE':'neg_mean_absolute_error',
                              'MAPE':'neg_mean_absolute_percentage_error'},
                   refit = 'RMSE',  # Refit the best parameters on the whole dataset
                   return_train_score= True,
                   verbose = 1,  # Add verbosity to show progress
                   )

logging.info('Start GridSearchCV...')
print("Starting GridSearchCV - this may take a while...")
grid = clf.fit(
    X_train, y_train
    )
logging.info('End GridSearchCV !')
#Store results
results = pd.DataFrame(grid.cv_results_)
best_para = grid.best_params_

msg = 'BEST Parameters: '+str(best_para)
logging.info(msg)

# Our final estimator
model = grid.best_estimator_

############
### Plot ###
############

colors = ['k', 'red', 'blue']

fig, axes = plt.subplots(1, 4, figsize = (15, 4))

xaxis , lines = parameters.keys()

for idx, score in enumerate(['RMSE', 'R2', 'MAE', 'MAPE']) :
    ax = axes[idx]

    for idx, md_value in enumerate(results['param_' + lines].unique()):
        sub = results[results['param_' + lines] == md_value]
        # test score plain line
        ax.plot(sub['param_' + xaxis], sub['mean_test_'+score], label = md_value, c = colors[idx])
        # train score dashed line
        ax.plot(sub['param_' + xaxis], sub['mean_train_'+score], ls = '--', c = colors[idx])

    # If max depth = None at some point
    if results['param_' + lines].isna().sum() :
        idx, md_value = 2, 'None'  
        sub = results[results['param_' + lines].isna()]
        ax.plot(sub['param_' + xaxis], sub['mean_test_'+score], label = md_value, c = colors[idx])
        ax.plot(sub['param_' + xaxis], sub['mean_train_'+score], ls = '--', c = colors[idx])
    
    ax.set_ylabel(score)
    ax.set_xlabel(xaxis)
    
# Avoid duplicated legends
plt.legend(title = lines)
    
plt.tight_layout()
plt.savefig('Contrails/Grid_Search_res_regression.png')
plt.close()

########################
### Final Evaluation ###
########################

#Train dataset predictions
y_train_preds = model.predict(X_train)
# test dataset prediction
y_test_preds = model.predict(X_test)

# Log the different scores
precision = 4

# Accuracy
msg = 'Training RMSE:' + str(round(root_mean_squared_error(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test RMSE:' + str(round(root_mean_squared_error(y_test, y_test_preds), precision))
logging.info(msg)
# F score
msg = 'Training R2:' + str(round(r2_score(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test R2:' + str(round(r2_score(y_test, y_test_preds), precision))
logging.info(msg)
# Precision
msg = 'Training MAE:' + str(round(mean_absolute_error(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test MAE:' + str(round(mean_absolute_error(y_test, y_test_preds), precision))
logging.info(msg)
# Recall
msg = 'Training MAPE:' + str(round(mean_absolute_percentage_error(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test MAPE:' + str(round(mean_absolute_percentage_error(y_test, y_test_preds), precision))
logging.info(msg)

#############################
### Feature importance
#############################

imp = pd.DataFrame(index = X_train.columns,
             data = model.feature_importances_,
             columns=['importance']).sort_values(by = 'importance', ascending = True)

plt.barh(imp.index, 100*imp.importance.values)
plt.xlabel('Sklearn feature importance (%)')

plt.tight_layout()
plt.savefig('Contrails/feature_importance_sklearn_regression.png',  bbox_inches='tight')

plt.close()

###################
### Plot impact ###
###################

#Rterieve info of test data
df = X_test.copy()
df['preds'] = y_test_preds
df['contrail_CO2_km'] = impact.loc[X_test.index, 'contrail_CO2_km']
df['total_flight_distance_km'] = impact.loc[X_test.index, 'total_flight_distance_km']

# Define highlight_feature for debugging
highlight_feature = 'night_score_bool_2'

# Plot
# Get the real impact of contrail per km - can be several minutes
x = np.sort(df.contrail_CO2_km)[::-1]
x_asc = np.sort(df.contrail_CO2_km)

# Debug checks - add after plotting variables are defined
print(f"Test dataset shape: {df.shape}")
print(f"Test dataset columns: {df.columns.tolist()}")
print(f"Unique values in {highlight_feature}: {df[highlight_feature].unique()}")
print(f"Count of values == 100: {(df[highlight_feature] == 100).sum()}")
print(f"Count of values != 100: {(df[highlight_feature] != 100).sum()}")
print(f"Sum of contrail_CO2_km: {np.sum(x_asc)}")
print(f"x_asc size: {x_asc.size}")

# Additional debugging for NaN issue
print(f"NaN in contrail_CO2_km: {df['contrail_CO2_km'].isna().sum()}")
print(f"Min total_flight_distance_km: {df['total_flight_distance_km'].min()}")
print(f"Max total_flight_distance_km: {df['total_flight_distance_km'].max()}")
print(f"Min contrail_CO2_km: {df['contrail_CO2_km'].min()}")
print(f"Max contrail_CO2_km: {df['contrail_CO2_km'].max()}")
print(f"Sample contrail_CO2_km values: {df['contrail_CO2_km'].head()}")

#< 0
plt.plot(np.linspace(0, 100, x.size)[np.where(x < 0)][::-1],
        (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc < 0)],
        c = 'blue',
        lw = 3,
        label = 'Cooling')

# = 0
plt.plot(np.linspace(0, 100, x.size)[np.where(x == 0)][::-1],
         (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc == 0)],
         c = 'grey',
         lw = 3,
         label = 'Neutral')

# > 0
plt.plot(np.linspace(0, 100, x.size)[np.where((x > 0) & (x < th))][::-1],
         (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where((x_asc > 0)  & (x_asc < th))],
         c = 'red',
         lw = 3,
         label = 'Warming')

# Highly warming
plt.plot(np.linspace(0, 100, x.size)[np.where(x >= th)][::-1],
         (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc >= th)],
         c = 'purple',
         lw = 3,
         label = 'Highly Warming')

##########################

# 0 line
plt.hlines(0, 0, 100,
           color = 'k',
           ls = '--')

# 0 line
plt.vlines(5, 0, 100,
           color = 'k',
           ls = '--')

# Random guess line
plt.plot([0, 100],
         [100, 0],
        c = 'gray',
        lw = 2,
        label = 'Uniform ordering'
)

# TRAIN
index = df.sort_values(by = ['preds', 'total_flight_distance_km']).index
x = df.loc[index, 'contrail_CO2_km']

plt.plot(np.linspace(0, 100, x.size)[::-1],
         (100*np.cumsum(x)/np.sum(x)),
        c = 'blue',
        lw = 2,
        label = 'Predictions'
)

# To distinguish 100% and the rest of the values
sub = df[[highlight_feature, 'total_flight_distance_km']].sort_values([highlight_feature, 'total_flight_distance_km'], ascending = [True, True])[highlight_feature]

# Add a curve that uses the highlight feature for sorting but doesn't separate values
x = df.contrail_CO2_km.loc[sub.index]
print(f"Orange curve - x length: {len(x)}")
print(f"Orange curve - x sum: {np.sum(x)}")
print(f"Orange curve - x_asc sum: {np.sum(x_asc)}")
print(f"Orange curve - first few values: {x.head()}")
print(f"Orange curve - last few values: {x.tail()}")

if len(x) > 0:
    y_values = (100*np.cumsum(x)/np.sum(x_asc))
    print(f"Orange curve - y_values range: {y_values.min()} to {y_values.max()}")
    print(f"Orange curve - y_values first few: {y_values[:5]}")
    print(f"Orange curve - y_values last few: {y_values[-5:]}")
    
    plt.plot(np.linspace(0, 100, x_asc.size)[::-1],
             y_values,
            c = 'orange',
            lw = 3,  # Make it thicker
            label = highlight_feature+'= all ns'
    )

x = df.contrail_CO2_km.loc[sub[sub==100].index]

# Only plot if there are values equal to 100
if len(x) > 0:
    plt.plot(np.linspace(0, 100, x_asc.size)[:x.size],
             (100*(np.sum(x_asc) - np.sum(x))/np.sum(x_asc) + (100*np.cumsum(x)/np.sum(x_asc)))[::-1],
            c = 'darkgreen',
            lw = 2,
            label = highlight_feature+'=100%'
    )

x = df.contrail_CO2_km.loc[sub[sub!=100].index]

# Only plot if there are values not equal to 100
if len(x) > 0:
    plt.plot(np.linspace(0, 100, x_asc.size)[-x.size:],
             (100*np.cumsum(x)/np.sum(x_asc))[::-1],
            c = 'seagreen',
            lw = 2,
            label = highlight_feature+'<100%'
    )

plt.legend(loc = 'upper right')
plt.xlabel('Flights(%)')
plt.ylabel('Cumulative Energy forcing per km (J/km)')

#Memory
x = None
x_asc = None

plt.savefig('Contrails/Impact_predictions_regression.png')



