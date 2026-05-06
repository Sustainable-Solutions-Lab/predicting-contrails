##################
### Librairies ###
##################

#Warning
import warnings
warnings.filterwarnings('ignore')

#Loggings
import logging
import sys

logging.basicConfig(filename='../results/contrail_avoidance_ml.txt', encoding='utf-8',
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
from xgboost import XGBClassifier
# Metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
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

# On wich year should we train the model
learning_year = 2019

if learning_year == 2019:
    test_year = 2021
elif learning_year == 2021 :
    test_year = 2019
else :
    logging.info('WRONG YEAR')

# Read new
df = pd.read_csv(path + str(learning_year) + '_prepared_shifts.csv')

msg = 'Number of flights: '+str(df.shape[0])
logging.info(msg)

msg = 'Size of dataset (Mo): ' + str(sys.getsizeof(df) >> 20)
logging.info(msg)

########################
### Encoding variables #
########################

encoding = True

if encoding :
    # Encoding time
    df['first_waypoint_time'] = pd.to_datetime(df['first_waypoint_time'])
    
    # target
    df['contrail_CO2_km'] = df['total_contrail_energy_forcing'] / df['total_flight_distance_km']
    
    # Time
    df['day_sin'] = df.first_waypoint_time.apply(lambda x : math.sin(np.pi * x.dayofyear * 2 / 365))
    df['day_cos'] = df.first_waypoint_time.apply(lambda x : math.cos(np.pi * x.dayofyear * 2 / 365))
    
    # Bearing
    df['bearing_sin'] = df.bearing.apply(lambda x : math.sin(np.pi * x * 2 / 360))
    df['bearing_cos'] = df.bearing.apply(lambda x : math.cos(np.pi * x * 2 / 360))
    
    #Longitude
    df['OriginLon_sin'] = df.OriginLon_bin.apply(lambda x : math.sin(np.pi * x * 2 / 360))
    df['OriginLon_cos'] = df.OriginLon_bin.apply(lambda x : math.cos(np.pi * x * 2 / 360))
    df['DestinationLon_sin'] = df.DestinationLon_bin.apply(lambda x : math.sin(np.pi * x * 2 / 360))
    df['DestinationLon_cos'] = df.DestinationLon_bin.apply(lambda x : math.cos(np.pi * x * 2 / 360))

    # Dummies
    df = pd.concat([
            df[[
                'contrail_CO2_km',
                'total_flight_distance_km',
                
                'day_sin', 'day_cos',
                # 'night_score', 
                # 'night_score_bool', 
                # 'night_score_full', 
                # 'night_score_rev',
                'land_score', 
                'bearing_sin','bearing_cos',
                
                #'OriginLon_bin','DestinationLon_bin',
                'OriginLon_sin','DestinationLon_sin',
                'OriginLon_cos','DestinationLon_cos',
                
                'OriginLat_bin','DestinationLat_bin'
                
        ]],
            df[
    ['night_score_full_'+str(shift) for shift in [0, 1, 2, 3]]
                ],
            df[
    ['night_score_bool_'+str(shift) for shift in [0, 1, 2, 3]]
                ]
            # pd.get_dummies(df.Origin_Region, prefix = 'Origin', dtype = int),
            # pd.get_dummies(df.Destination_Region, prefix = 'Destination', dtype = int)
        ], 
                axis = 1)

# Threshold definition - 10% worst flights
th = df.contrail_CO2_km.quantile(.95)

# Encoding categorical
df.loc[df.contrail_CO2_km <= th, 'contrail_type'] = 0
df.loc[df.contrail_CO2_km > th, 'contrail_type'] = 1


msg = 'Size of dataset after feature selection (Mo): ' + str(sys.getsizeof(df) >> 20)
logging.info(msg)

# Types
df = df.astype({
    'day_sin': 'float16',
    'day_cos': 'float16',
    'contrail_CO2_km': 'float64',
    'contrail_type': 'int16',
    'total_flight_distance_km': 'float64',
    # 'night_score': 'float16',
    # 'night_score_bool': 'float16',
    # 'night_score_full': 'float16',
    # 'night_score_rev': 'float16',
    'land_score': 'float16',
    'bearing_sin': 'float16',
    'bearing_cos': 'float16',
    # 'OriginLon_bin' : 'float16',
    # 'DestinationLon_bin' : 'float16',
    'OriginLon_sin' : 'float16',
    'DestinationLon_sin' : 'float16',
    'OriginLon_cos' : 'float16',
    'DestinationLon_cos' : 'float16',
    
    'OriginLat_bin': 'float16',
    'DestinationLat_bin': 'float16',
    
    # OLD regions reaftures
    # 'Origin_Africa':'int16',
    # 'Origin_Asia': 'int16',
    # 'Origin_Europe':'int16',
    # 'Origin_Middle East': 'int16',
    # 'Origin_North America': 'int16',
    # 'Origin_Oceania': 'int16',
    # 'Origin_Russia': 'int16',
    # 'Origin_South America':'int16',
    # 'Destination_Africa': 'int16',
    # 'Destination_Asia':'int16',
    # 'Destination_Europe': 'int16',
    # 'Destination_Middle East': 'int16',
    # 'Destination_North America': 'int16',
    # 'Destination_Oceania': 'int16',
    # 'Destination_Russia':'int16',
    # 'Destination_South America': 'int16'
})

msg = 'Size of dataset after type conversion (Mo): ' + str(sys.getsizeof(df) >> 20)
logging.info(msg)

msg = str(df.columns)
logging.info(msg)

################################################
#### Impact of night score features ############
################################################
# Plot
# Get the real impact of contrail per km - can be several minutes
x = np.sort(df.contrail_CO2_km)[::-1]
x_asc = np.sort(df.contrail_CO2_km)

#< 0
plt.plot(np.linspace(0, 100, x.size)[np.where(x < 0)][::-1],
        (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc < 0)],
        c = 'blue',
        lw = 3,
        label = 'Cooling')

# = 0
plt.plot(np.linspace(0, 100, x.size)[np.where(x == 0)][::-1],
         (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc == 0)],
         c = 'lightgray',
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

# Simple rule with night scores
########################################

highlight_feature = 'night_score_bool_1'

for idx, highlight_feature in enumerate([
     'night_score_bool_0',
      'night_score_bool_1',
       'night_score_bool_2',
        'night_score_bool_3',
         'night_score_full_0',
         'night_score_full_1',
          'night_score_full_2',
          'night_score_full_3'
]):
    if idx < 4:
        colors = [ 'darkgoldenrod', 'goldenrod', 'gold', 'khaki']
        sorting = True
    else :
        colors = [ 'darkgreen', 'green', 'limegreen', 'lime']
        sorting = False
        idx -= 4
    sub = df[[highlight_feature, 'total_flight_distance_km']].sort_values([highlight_feature, 'total_flight_distance_km'],
                                                                          ascending = [sorting, True] # False if full score
                                                                          )[highlight_feature]
    #df.night_score.sort_values(ascending = True)
    x = df.contrail_CO2_km.loc[sub.index]

    plt.plot(np.linspace(0, 100, x.size)[::-1],
            100*np.cumsum(x)/np.sum(x),
            c = colors[idx],
            lw = 2,
            alpha = .8,
            label = highlight_feature
    )


plt.legend(loc = 'upper right', fontsize = 6)
plt.xlabel('% of flights')
plt.ylabel('Cumulated % of the total RF/km')

#Memory
x = None
x_asc = None

plt.savefig('Contrails/Impact_predictions_features_'+str(learning_year)+'.svg')
plt.close()

####################################################################

# Features and variables
target_variable = 'contrail_type'

# Features
X_train = df.drop([target_variable, 'contrail_CO2_km', 'total_flight_distance_km'], axis = 1)
# Labels
y_train = df[target_variable]

# for later: impact in terms of radiative effect per km - could be absolute too
impact = df[['contrail_CO2_km', 'total_flight_distance_km']]

# save some RAM
df = None

####################################
### Train all and predict on other year
#################################
XGB_coeff = y_train.shape[0] / y_train.sum()

model = XGBClassifier(
                    n_estimators=1000, 
                    max_depth=10, 
                    n_jobs = -1,
                    learning_rate=.5, 
                    # objective='binary:logistic',
                    scale_pos_weight=XGB_coeff
)

# Fit on the entire year
model.fit(X_train, y_train)

# Read new
df = pd.read_csv(path + str(test_year) +  '_prepared_shifts.csv')

#### Test
df['first_waypoint_time'] = pd.to_datetime(df['first_waypoint_time'])

# target
df['contrail_CO2_km'] = df['total_contrail_energy_forcing'] / df['total_flight_distance_km']

# Time
df['day_sin'] = df.first_waypoint_time.apply(lambda x : math.sin(np.pi * x.dayofyear * 2 / 365))
df['day_cos'] = df.first_waypoint_time.apply(lambda x : math.cos(np.pi * x.dayofyear * 2 / 365))

# Bearing
df['bearing_sin'] = df.bearing.apply(lambda x : math.sin(np.pi * x * 2 / 360))
df['bearing_cos'] = df.bearing.apply(lambda x : math.cos(np.pi * x * 2 / 360))

#Longitude
df['OriginLon_sin'] = df.OriginLon_bin.apply(lambda x : math.sin(np.pi * x * 2 / 360))
df['OriginLon_cos'] = df.OriginLon_bin.apply(lambda x : math.cos(np.pi * x * 2 / 360))
df['DestinationLon_sin'] = df.DestinationLon_bin.apply(lambda x : math.sin(np.pi * x * 2 / 360))
df['DestinationLon_cos'] = df.DestinationLon_bin.apply(lambda x : math.cos(np.pi * x * 2 / 360))

# Dummies
df = pd.concat([
        df[[
            'contrail_CO2_km',
            'total_flight_distance_km',
            
            'day_sin', 'day_cos',
            # 'night_score', 
            # 'night_score_bool', 
            # 'night_score_full', 
            # 'night_score_rev',
            'land_score', 
            'bearing_sin','bearing_cos',
            
            #'OriginLon_bin','DestinationLon_bin',
            'OriginLon_sin','DestinationLon_sin',
            'OriginLon_cos','DestinationLon_cos',
            
            'OriginLat_bin','DestinationLat_bin'
            
    ]],
        df[
['night_score_full_'+str(shift) for shift in [0, 1, 2, 3]]
            ],
        df[
['night_score_bool_'+str(shift) for shift in [0, 1, 2, 3]]
            ]
        # pd.get_dummies(df.Origin_Region, prefix = 'Origin', dtype = int),
        # pd.get_dummies(df.Destination_Region, prefix = 'Destination', dtype = int)
    ], 
            axis = 1)

# Threshold definition - 10% worst flights
th = df.contrail_CO2_km.quantile(.95)

# Encoding categorical
df.loc[df.contrail_CO2_km <= th, 'contrail_type'] = 0
df.loc[df.contrail_CO2_km > th, 'contrail_type'] = 1


msg = 'Size of dataset after feature selection (Mo): ' + str(sys.getsizeof(df) >> 20)
logging.info(msg)

# Types
df = df.astype({
'day_sin': 'float16',
'day_cos': 'float16',
'contrail_CO2_km': 'float64',
'contrail_type': 'int16',
'total_flight_distance_km': 'float64',
# 'night_score': 'float16',
# 'night_score_bool': 'float16',
# 'night_score_full': 'float16',
# 'night_score_rev': 'float16',
'land_score': 'float16',
'bearing_sin': 'float16',
'bearing_cos': 'float16',
# 'OriginLon_bin' : 'float16',
# 'DestinationLon_bin' : 'float16',
'OriginLon_sin' : 'float16',
'DestinationLon_sin' : 'float16',
'OriginLon_cos' : 'float16',
'DestinationLon_cos' : 'float16',

'OriginLat_bin': 'float16',
'DestinationLat_bin': 'float16',

})

# Predictionq
y_train_preds = model.predict(X_train)
y_test_preds = model.predict(df.drop([target_variable, 'contrail_CO2_km', 'total_flight_distance_km'], axis = 1))
y_test = df[target_variable]

msg = 'Training on whole '+str(learning_year)+', Testing on : '+str(test_year)
logging.info('FOR Full YEAR : ' + str(test_year))

# Log the different scores
precision = 4
# Accuracy
msg = 'Training Accuracy:' + str(round(accuracy_score(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test Accuracy:' + str(round(accuracy_score(y_test, y_test_preds), precision))
logging.info(msg)
# F score
msg = 'Training F1-Score:' + str(round(f1_score(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test F1-Score:' + str(round(f1_score(y_test, y_test_preds), precision))
logging.info(msg)
# Precision
msg = 'Training Precision:' + str(round(precision_score(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test Precision:' + str(round(precision_score(y_test, y_test_preds), precision))
logging.info(msg)
# Recall
msg = 'Training Recall:' + str(round(recall_score(y_train, y_train_preds ), precision))
logging.info(msg)
msg = 'Test Recall:' + str(round(recall_score(y_test, y_test_preds), precision))
logging.info(msg)

# Feature importance
imp = pd.DataFrame(index = X_train.columns,
             data = model.feature_importances_,
             columns=['importance']).sort_values(by = 'importance', ascending = True)

plt.barh(imp.index, 100*imp.importance.values)
plt.xlabel('Sklearn feature importance (%)')

plt.tight_layout()
plt.savefig('Contrails/feature_importance_sklearn-'+str(learning_year)+'.png',  bbox_inches='tight')
plt.close()

###################
### Plot impact ###
###################

#Rterieve info of test data
df['preds'] = y_test_preds

# Plot
# Get the real impact of contrail per km - can be several minutes
x = np.sort(df.contrail_CO2_km)[::-1]
x_asc = np.sort(df.contrail_CO2_km)

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

# ML model
index = df.sort_values(by = ['preds', 'total_flight_distance_km']).index
x = df.loc[index, 'contrail_CO2_km']

plt.plot(np.linspace(0, 100, x.size)[:int(df.preds.sum())][::-1],
         (100*np.cumsum(x)/np.sum(x))[-int(df.preds.sum()):],
        c = 'darkorange',
        lw = 2,
        label = 'Predictions - highly warming'
)

plt.plot(np.linspace(0, 100, x.size)[int(df.preds.sum()):][::-1],
         (100*np.cumsum(x)/np.sum(x))[:-int(df.preds.sum())],
        c = 'deepskyblue',
        lw = 2,
        label = 'Predictions - others'
)

# Simple rule with night scores
#######################################
# Show the impacts if we sort only by this specific feature

highlight_feature = 'night_score_full_2'

# sub = df[['night_score', 'total_flight_distance_km']].sort_values([highlight_feature, 'total_flight_distance_km'], ascending = [True, True])[highlight_feature]
# #df.night_score.sort_values(ascending = True)
# x = df.contrail_CO2_km.loc[sub.index]

# plt.plot(np.linspace(0, 100, x.size)[::-1],
#          100*np.cumsum(x)/np.sum(x),
#         c = 'mediumblue',
#         lw = 2,
#         label = highlight_feature
# )

# To distinguish 100% and the rest of teh values
# Maybe full should be ascending = False compared to bool
sub = df[[highlight_feature, 'total_flight_distance_km']].sort_values([highlight_feature, 'total_flight_distance_km'], ascending = [False, True])[highlight_feature]
x = df.contrail_CO2_km.loc[sub[sub==100].index]

plt.plot(np.linspace(0, 100, x_asc.size)[:x.size],
         (100*(np.sum(x_asc) - np.sum(x))/np.sum(x_asc) + (100*np.cumsum(x)/np.sum(x_asc)))[::-1],
        c = 'darkgreen',
        lw = 2,
        label = highlight_feature+'=100%'
)

x = df.contrail_CO2_km.loc[sub[sub!=100].index]

plt.plot(np.linspace(0, 100, x_asc.size)[-x.size:],
         (100*np.cumsum(x)/np.sum(x_asc))[::-1],
        c = 'seagreen',
        lw = 2,
        label = highlight_feature+'<100%'
)


plt.legend(loc = 'upper right')
plt.xlabel('% of flights')
plt.ylabel('Cumulated % of the total RF/km')
plt.savefig('Contrails/Impact_predictions_'+str(learning_year)+'_on_'+str(test_year)+'.png')
plt.close()
#Memory
x = None
x_asc = None
df = None


logging.info('STOP')



# Previous part with training / testing on the same year:

########################
### Train/Test Split ### Learning year ########################################################################
########################

# # We define training and test set
# X_train, X_test, y_train, y_test = train_test_split(
#     #X, y,
#     X_train, y_train,
#     random_state = 42,
#     test_size = 0.3, # How many % are attributed to test ? 
#     stratify = y_train #y # Make sure we have the same rate of class 1
#     )

# # X_test and y_test are only use for final validation

# # Memory
# #X, y = None, None

# ###################################
# ### Handling imbalanced dataset ###
# ###################################

# msg = 'Training label repartition : '+str(Counter(y_train))
# logging.info(msg)

# technique = 'Coeffs'
# # 'random_under_sampling'
# # 'SMOTE'
# # 'Coeffs'
# # 'SMOTEENN' Very long - needs more than +12 hours

# msg = 'Imbalanced technique handling: ' + technique
# logging.info(msg)

# if technique == 'SMOTE':
#     smt = SMOTE(
#             n_jobs = -1, 
#             k_neighbors = 3,
#             random_state = 42
# )
#     X_train, y_train = smt.fit_resample(X_train, y_train)
    
# elif technique == 'SMOTEENN':
#     # Hybridation technique with undersampling after it
#     smtenn = SMOTEENN(
#         random_state = 42,
#         smote = SMOTE(k_neighbors = 3),
#         enn = EditedNearestNeighbours(n_neighbors=2, sampling_strategy='all'),
#         n_jobs = -1
# )
#     X_train, y_train = smtenn.fit_resample(X_train, y_train)
    
# elif technique == 'random_under_sampling':
#     # Get a balanced training set
#     y_train = pd.concat([
#         y_train.iloc[np.where(y_train == 1)].sample( n = int(y_train.sum())), 
#         # We get the same amount of flight for each category
#         y_train.iloc[np.where(y_train == 0)].sample( n = int(y_train.sum())),

#     ])
#     #Select correspondig features
#     X_train = X_train.loc[y_train.index]
    
# # Model coefficients
# RF, XGB_coeff = None, 1
# if technique == 'Coeffs':
#     RF, XGB_coeff = 'balanced', y_train.shape[0] / y_train.sum()
    
# msg = 'AFTER Imbalanced Treatment : '+str(Counter(y_train))
# logging.info(msg)

# ######################
# ### Grid Search CV ###
# ######################


# # Model
# # model = RandomForestClassifier(
# #     # n_estimators = 5, # Number of trees
# #     # max_depth = 15, #Maxh depth of a tree
# #     n_jobs = -1, #Number of CPUs to use, -1 = max number allowed by the environment
# #     # Take into account the unbalance dataset
# #    class_weight = RF
# # )

# model = XGBClassifier(
#                     # n_estimators=1000, 
#                     # max_depth=8, 
#                     n_jobs = -1,
#                     learning_rate=.5, 
#                     # objective='binary:logistic',
#                     scale_pos_weight=XGB_coeff
# )

# # Wich paramaters do we want to try
# parameters = {
#              'n_estimators' : [10, 1000],
#             #   'learning_rate' : [.3, .5, .8],
#              # 'mean_child_weight' : [1, 5, 10],
#              # 'lambda' : [1, 10, 20]
#               #'gamma' : [0, 1, 10]
#               'max_depth' : [5, 10],
#              # 'scale_pos_weight' : [5, 10, 20]
#               }

# # Define the scores and the cv
# clf = GridSearchCV(model, parameters, 
#                    cv = 3, 
#                    scoring = {'Accuracy':'accuracy',
#                               'F1-score':'f1', 
#                               'Precision':'precision',
#                               'Recall':'recall'},
#                    refit = 'F1-score',  # Refit the best parameters on the whole dataset
#                    return_train_score= True,
#                   # verbose = 10 # get some details --> Hidden the results that we need...
#                    )

# logging.info('Start GridSearchCV...')
# grid = clf.fit(
#     X_train, y_train
#     )
# logging.info('End GridSearchCV !')
# #Store results
# results = pd.DataFrame(grid.cv_results_)
# best_para = grid.best_params_

# msg = 'BEST Parameters: '+str(best_para)
# logging.info(msg)

# # Our final estimator
# model = grid.best_estimator_

# ############
# ### Plot ###
# ############

# colors = ['k', 'red', 'blue']

# fig, axes = plt.subplots(1, 4, figsize = (15, 4))

# xaxis , lines = parameters.keys()

# for idx, score in enumerate(['Accuracy', 'F1-score', 'Precision', 'Recall']) :
#     ax = axes[idx]

#     for idx, md_value in enumerate(results['param_' + lines].unique()):
#         sub = results[results['param_' + lines] == md_value]
#         # test score plain line
#         ax.plot(sub['param_' + xaxis], sub['mean_test_'+score], label = md_value, c = colors[idx])
#         # train score dashed line
#         ax.plot(sub['param_' + xaxis], sub['mean_train_'+score], ls = '--', c = colors[idx])

#     # If max depth = None at some point
#     if results['param_' + lines].isna().sum() :
#         idx, md_value = 2, 'None'  
#         sub = results[results['param_' + lines].isna()]
#         ax.plot(sub['param_' + xaxis], sub['mean_test_'+score], label = md_value, c = colors[idx])
#         ax.plot(sub['param_' + xaxis], sub['mean_train_'+score], ls = '--', c = colors[idx])
    
#     #ax.set_ylim(0.2, 0.8)
#     ax.set_ylabel(score)
#     ax.set_xlabel(xaxis)
    
# # Avoid duplicated legends
# plt.legend(title = lines)
    
# # axes[0].set_ylim(.7, 1)   
# # axes[1].set_ylim(.2, .5)  
# plt.tight_layout()
# plt.savefig('Contrails/Grid_Search_res.png')
# plt.close()

# ########################
# ### Final Evaluation ###
# ########################

# #Train dataset predictions
# y_train_preds = model.predict(X_train)
# # test dataset prediction
# y_test_preds = model.predict(X_test)

# # Log the different scores
# precision = 4
# # Accuracy
# msg = 'Training Accuracy:' + str(round(accuracy_score(y_train, y_train_preds ), precision))
# logging.info(msg)
# msg = 'Test Accuracy:' + str(round(accuracy_score(y_test, y_test_preds), precision))
# logging.info(msg)
# # F score
# msg = 'Training F1-Score:' + str(round(f1_score(y_train, y_train_preds ), precision))
# logging.info(msg)
# msg = 'Test F1-Score:' + str(round(f1_score(y_test, y_test_preds), precision))
# logging.info(msg)
# # Precision
# msg = 'Training Precision:' + str(round(precision_score(y_train, y_train_preds ), precision))
# logging.info(msg)
# msg = 'Test Precsion:' + str(round(precision_score(y_test, y_test_preds), precision))
# logging.info(msg)
# # Recall
# msg = 'Training Recall:' + str(round(recall_score(y_train, y_train_preds ), precision))
# logging.info(msg)
# msg = 'Test Recall:' + str(round(recall_score(y_test, y_test_preds), precision))
# logging.info(msg)

# #############################
# ### Feature importance
# #############################

# imp = pd.DataFrame(index = X_train.columns,
#              data = model.feature_importances_,
#              columns=['importance']).sort_values(by = 'importance', ascending = True)

# plt.barh(imp.index, 100*imp.importance.values)
# plt.xlabel('Sklearn feature importance (%)')

# plt.tight_layout()
# plt.savefig('Contrails/feature_importance_sklearn.png',  bbox_inches='tight')
# plt.close()

# # Confusion matrices
# # conf_matrix_train = confusion_matrix(y_train, y_train_preds, normalize='true' )
# # conf_matrix_test = confusion_matrix(y_test, y_test_preds, normalize='true' )

# # l1=[('Prediction', 'Normal'),  ('Prediction', 'Warming')]
# # l2=[('Truth', 'Normal'),  ('Truth', 'Warming')]

# # msg = str(pd.DataFrame(data = conf_matrix_test,
# #              columns = pd.MultiIndex.from_tuples(l1),
# #              index = pd.MultiIndex.from_tuples(l2)).style.background_gradient(
# #                  'coolwarm', axis=0, vmin=0, vmax=1
# #                  ).format(precision = 3))
# # logging.info(msg)

# ###################
# ### Plot impact ###
# ###################

# #Rterieve info of test data
# df = X_test.copy()
# df['preds'] = y_test_preds
# df['contrail_CO2_km'] = impact.loc[X_test.index, 'contrail_CO2_km']
# df['total_flight_distance_km'] = impact.loc[X_test.index, 'total_flight_distance_km']

# # Plot
# # Get the real impact of contrail per km - can be several minutes
# x = np.sort(df.contrail_CO2_km)[::-1]
# x_asc = np.sort(df.contrail_CO2_km)

# #< 0
# plt.plot(np.linspace(0, 100, x.size)[np.where(x < 0)][::-1],
#         (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc < 0)],
#         c = 'blue',
#         lw = 3,
#         label = 'Cooling')

# # = 0
# plt.plot(np.linspace(0, 100, x.size)[np.where(x == 0)][::-1],
#          (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc == 0)],
#          c = 'grey',
#          lw = 3,
#          label = 'Neutral')

# # > 0
# plt.plot(np.linspace(0, 100, x.size)[np.where((x > 0) & (x < th))][::-1],
#          (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where((x_asc > 0)  & (x_asc < th))],
#          c = 'red',
#          lw = 3,
#          label = 'Warming')

# # Highly warming
# plt.plot(np.linspace(0, 100, x.size)[np.where(x >= th)][::-1],
#          (100*np.cumsum(x_asc)/np.sum(x_asc))[np.where(x_asc >= th)],
#          c = 'purple',
#          lw = 3,
#          label = 'Highly Warming')

# ##########################

# # 0 line
# plt.hlines(0, 0, 100,
#            color = 'k',
#            ls = '--')

# # 0 line
# plt.vlines(5, 0, 100,
#            color = 'k',
#            ls = '--')

# # Random guess line
# plt.plot([0, 100],
#          [100, 0],
#         c = 'gray',
#         lw = 2,
#         label = 'Uniform ordering'
# )

# # TRAIN
# index = df.sort_values(by = ['preds', 'total_flight_distance_km']).index
# x = df.loc[index, 'contrail_CO2_km']

# plt.plot(np.linspace(0, 100, x.size)[:int(df.preds.sum())][::-1],
#          (100*np.cumsum(x)/np.sum(x))[-int(df.preds.sum()):],
#         c = 'darkorange',
#         lw = 2,
#         label = 'Predictions - highly warming'
# )

# plt.plot(np.linspace(0, 100, x.size)[int(df.preds.sum()):][::-1],
#          (100*np.cumsum(x)/np.sum(x))[:-int(df.preds.sum())],
#         c = 'deepskyblue',
#         lw = 2,
#         label = 'Predictions - others'
# )


# # Simple rule with night scores
# ########################################

# # sub = df[['night_score', 'total_flight_distance_km']].sort_values(['night_score', 'total_flight_distance_km'], ascending = [True, True]).night_score
# # #df.night_score.sort_values(ascending = True)
# # x = df.contrail_CO2_km.loc[sub.index]

# # plt.plot(np.linspace(0, 100, x.size)[::-1],
# #          100*np.cumsum(x)/np.sum(x),
# #         c = 'mediumblue',
# #         lw = 2,
# #         label = 'Night score sorting'
# # )

# # sub = df[['night_score_bool', 'total_flight_distance_km']].sort_values(['night_score_bool', 'total_flight_distance_km'], ascending = [True, True]).night_score_bool
# # #df.night_score_bool.sort_values(ascending = True)
# # x = df.contrail_CO2_km.loc[sub.index]

# # plt.plot(np.linspace(0, 100, x.size)[::-1],
# #          100*np.cumsum(x)/np.sum(x),
# #         c = 'darkblue',
# #         lw = 2,
# #         label = 'Night score boolean sorting'
# # )


# plt.legend(loc = 'upper right')
# plt.xlabel('% of flights')
# plt.ylabel('Cumulated % of the total RF/km')

# #Memory
# x = None
# x_asc = None

# plt.savefig('Contrails/Impact_predictions.png')