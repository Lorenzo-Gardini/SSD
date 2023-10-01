# IMPORTS
from itertools import product

import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.stats.weightstats

# XGBoost
from xgboost import XGBRegressor
import threading
# statsmodels
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor)
# utility functions
from fourier_regressor import FourierRegressor
from utility_functions import *
# nn builder functions
import keras
from keras_utility_functions import *
from sklearn_utility_functions import *

# -------------------- DATAFRAME --------------------
df = pd.read_csv('res/mitv.csv')

"""
The data set includes hourly Minneapolis-St Paul traffic volume for westbound I-94 from 2012-2018. 
It contains 48204 instances with 9 attributes:

- holiday: categorical US national holidays plus regional holiday;
- temp: Numeric average temperature in kelvin;
- rain_1h: numeric amount in mm of rain that occurred in the hour;
- snow_1h: numeric amount in mm of snow that occurred in the hour;
- clouds_all: numeric percentage of cloud cover;
- weather_main: categorical short textual description of the current weather;
- weather_description: Categorical longer textual description of the current weather;
- date_time: date and hour of the data collected in local CST time;
- traffic_volume: Numeric hourly I-94 ATR 301 reported westbound traffic volume.
"""

print(df)
# -------------------- BASE PREPROCESSING --------------------

pred_y = 'traffic_volume'

# -------------------- LABEL ENCODING --------------------

print(df.info())

# as there are many values for each of this columns label encoding is better
for col in ['holiday', 'weather_main', 'weather_description']:
    print(f'{col} : {df[col].unique().size}')

for col in ['holiday', 'weather_main', 'weather_description']:
    df[col] = df[col].astype('category').cat.codes

print(df)

# -------------------- DATETIME MANAGEMENT --------------------

values = df['date_time'].value_counts()
values[values > 1].count()

df = df.drop_duplicates('date_time', keep='last')
df.index = pd.DatetimeIndex(df['date_time'])
df = df.sort_index()
df = df.resample('H').first()
df.isna().sum()
plt.plot(df[pred_y])
plt.title('Dataset with dates')
plt.show()
# -------------------- FILL VALUES --------------------

df = df.interpolate()
df['date_time'] = df.index
print(df.isna().sum())
plt.plot(df[pred_y])
plt.title('Dataset with interpolation')
plt.show()
# -------------------- CYCLIC TIME ENCODING --------------------

df['hour_sin'], df['hour_cos'] = cycles(df.index.hour, 23)
df['day_week_sin'], df['day_week_cos'] = cycles(df.index.dayofweek, 7)
df['month_sin'], df['month_cos'] = cycles(df.index.month, 12)

df.drop(columns='date_time', inplace=True)
print(df)

# -------------------- ADD LAGS --------------------

lags = 24 * 7  # one week

lagged_columns = [df[pred_y].diff(lag).rename(f'{pred_y}_lag_{lag}') for lag in range(1, lags + 1)]

df = pd.concat([df, *lagged_columns], axis=1)

df = df.dropna()
print(df)

# -------------------- CUT --------------------

cut_df = df['2016':]
plt.plot(cut_df[pred_y])
plt.title('Dataset from 2016')
plt.show()
# -------------------- PLOTS --------------------

cut_df.groupby(by=[cut_df.index.year]).boxplot(column=[pred_y], subplots=False)
plt.title(f'Boxplot of {pred_y} group by year')
plt.show()
# -------------------- AUGMENTED DICKEY-FULLER TEST --------------------

test_results = adfuller(cut_df[pred_y])
print('ADF Statistic: %f' % test_results[0])
print('p-value: %f' % test_results[1])
for key, value in test_results[4].items():
    print(f'{key}: {value}')

# -------------------- DIVIDE X AND Y --------------------

x = cut_df.drop(columns=pred_y)
y = cut_df[pred_y]

# -------------------- SPLIT TRAIN VALIDATION TEST --------------------

full_train_x, test_x, full_train_y, test_y = take_last(x, y, quantity=2, what='week')
train_x, validation_x, train_y, validation_y = split_continuously(full_train_x, full_train_y, partitions=[0.75, 0.25])

print(f'Dataset: {len(y)}')
print(f'Full trainset: {len(full_train_y)}')
print(f'testset: {len(test_y)}')

print(f'trainset: {len(train_y)}')
print(f'validation: {len(validation_y)}')

plots([full_train_y, test_y], ['full train', 'test'])
plots([train_y, validation_y, test_y], ['train', 'validation', 'test'])

# -------------------- NORMALIZATION --------------------

# normalize x data
scaler_x, n_full_train_x, n_train_x, n_validation_x, n_test_x = normalize_2d(full_train_x,
                                                                             full_train_x,
                                                                             train_x,
                                                                             validation_x,
                                                                             test_x,
                                                                             transform_function=MinMaxScaler())

# normalize y data
scaler_y, n_full_train_y, n_train_y, n_validation_y, n_test_y = normalize_1d(full_train_y,
                                                                             full_train_y,
                                                                             train_y,
                                                                             validation_y,
                                                                             test_y,
                                                                             transform_function=MinMaxScaler())

# -------------------- SEED --------------------

np.random.seed(1234)

# -------------------- MODELS --------------------

do_grid_search = False
do_train = False
models_path = 'res/'
models = {}
forecasts = {}

# -------------------- NON NEURAL MODELS --------------------

# -------------------- FOURIER --------------------

one_week = 24 * 7

if do_train:
    models['FourierRegressor'] = FourierRegressor(h_harm=1000).fit(signal=n_full_train_y[-one_week:])
    save_model(models['FourierRegressor'], models_path + 'FourierRegressor')
else:
    models['FourierRegressor'] = load_model(models_path + 'FourierRegressor')

forecasts['FourierRegressor'] = endog_forecast(models['FourierRegressor'], test_y, scaler_y)

# -------------------- SARIMAX --------------------
if do_grid_search:
    print('Grid search on SARIMAX')
    models['SARIMAX'] = auto_arima(n_full_train_y, n_full_train_x,
                                   start_p=0, max_p=2,
                                   max_d=2,
                                   start_q=0, max_q=2,
                                   m=1, stationary=True,
                                   n_jobs=threading.active_count(), trace=2, suppress_warnings=False)
elif do_train:
    print('Training SARIMAX')
    models['SARIMAX'] = SARIMAX(n_full_train_y, exog=n_full_train_x, order=(1, 1, 0)).fit()
    save_model(models['SARIMAX'], models_path + 'SARIMAX')
else:
    models['SARIMAX'] = load_model(models_path + 'SARIMAX')

models['SARIMAX'].plot_diagnostics()
plt.show()

forecasts['SARIMAX'] = endog_forecast(models['SARIMAX'], test_y, scaler_y, exog=n_test_x)

# -------------------- HWES --------------------
if do_train:
    print('Training HWES')
    models['HWES'] = ExponentialSmoothing(n_full_train_y, trend=None,
                                          seasonal_periods=24 * 7,
                                          seasonal='add',
                                          initialization_method="heuristic").fit(optimized=True)
    save_model(models['HWES'], models_path + 'HWES')
else:
    models['HWES'] = load_model(models_path + 'HWES')

forecasts['HWES'] = endog_forecast(models['HWES'], test_y, scaler_y)

# -------------------- LINEAR REGRESSION --------------------

if do_grid_search:
    print('Grid search on LinearRegression')
    LR_grid = {
        'fit_intercept': [False, True]
    }
    models['LinearRegression'] = grid_search(LinearRegression(fit_intercept=False), LR_grid, n_full_train_x,
                                             n_full_train_y)

elif do_train:
    print('Training LinearRegression')
    models['LinearRegression'] = LinearRegression(fit_intercept=False).fit(n_full_train_x, n_full_train_y)
    save_model(models['LinearRegression'], models_path + 'LinearRegression')
else:
    models['LinearRegression'] = load_model(models_path + 'LinearRegression')

forecasts['LinearRegression'] = forecast(models['LinearRegression'], n_test_x, test_y.index, scaler_y)
# -------------------- SVR --------------------

if do_grid_search:
    print('Grid search on SVR')
    SVR_param_grid = {
        'C': np.arange(100, 400, 100),
        'gamma': [0.15],
        'epsilon': [0.01]
    }

    models['SVR'] = grid_search(SVR(), SVR_param_grid, n_full_train_x, n_full_train_y)

elif do_train:
    print('Training SVR')
    models['SVR'] = SVR(C=100, epsilon=0.01, gamma=0.15).fit(n_full_train_x, n_full_train_y)
    save_model(models['SVR'], models_path + 'SVR')
else:
    models['SVR'] = load_model(models_path + 'SVR')

forecasts['SVR'] = forecast(models['SVR'], n_test_x, test_y.index, scaler_y)
# -------------------- RANDOM FOREST REGRESSOR --------------------

if do_grid_search:
    print('Grid search on RandomForest')
    RFR_param_grid = {'max_depth': [80],
                      'max_features': ['sqrt'],
                      'min_samples_leaf': [1],
                      'min_samples_split': [2],
                      'n_estimators': [900]
                      }

    models['RandomForest'] = grid_search(RandomForestRegressor(), RFR_param_grid, n_full_train_x, n_full_train_y)

elif do_train:
    print('Training RandomForest')
    models['RandomForest'] = RandomForestRegressor(max_depth=80, max_features='sqrt', n_estimators=900).fit(
        n_full_train_x,
        n_full_train_y)
    save_model(models['RandomForest'], models_path + 'RandomForest')
else:
    models['RandomForest'] = load_model(models_path + 'RandomForest')

forecasts['RandomForest'] = forecast(models['RandomForest'], n_test_x, test_y.index, scaler_y)
# -------------------- ADABOOST REGRESSOR --------------------

if do_grid_search:
    print('Grid search on AdaBoost')
    ABR_param_grid = {
        'base_estimator': [DecisionTreeRegressor(max_depth=80, max_features='sqrt')],
        'n_estimators': [70, 80, 90, 100],

    }
    models['AdaBoost'] = grid_search(AdaBoostRegressor(), ABR_param_grid, n_full_train_x, n_full_train_y)

elif do_train:
    print('Training AdaBoost')
    models['AdaBoost'] = AdaBoostRegressor(DecisionTreeRegressor(max_depth=80, max_features='sqrt'),
                                           n_estimators=90).fit(n_full_train_x, n_full_train_y)
    save_model(models['AdaBoost'], models_path + 'AdaBoost')
else:
    models['AdaBoost'] = load_model(models_path + 'AdaBoost')

forecasts['AdaBoost'] = forecast(models['AdaBoost'], n_test_x, test_y.index, scaler_y)
# -------------------- XGBOOST --------------------

if do_grid_search:
    print('Grid search on XGBoost')
    XGB_param_grid = {'gamma': [0, 0.01],
                      'learning_rate': [0.1, 0.5],
                      'max_depth': [50, 80],
                      'n_estimators': [150, 200],
                      'reg_alpha': [0, 0.01],
                      'reg_lambda': [1, 10]}

    models['XGBoost'] = grid_search(XGBRegressor(objective='reg:squarederror'),
                                    XGB_param_grid,
                                    n_full_train_x,
                                    n_full_train_y)

elif do_train:
    print('Training XGBoost')
    models['XGBoost'] = XGBRegressor(objective='reg:squarederror', max_depth=50, n_estimators=150).fit(n_full_train_x,
                                                                                                       n_full_train_y)
    save_model(models['XGBoost'], models_path + 'XGBoost')
else:
    models['XGBoost'] = load_model(models_path + 'XGBoost')

forecasts['XGBoost'] = forecast(models['XGBoost'], n_test_x, test_y.index, scaler_y)
# -------------------- NEURAL MODELS --------------------

epoch_count = 100
patience = 5
batch_size = 32
features = n_train_x.shape[1]

# -------------------- MLP --------------------
if do_train:
    print('Training MLP')
    models['MLP'] = build_deep_mlp(features)
    models['MLP'], _ = keras_fit(models['MLP'], n_train_x, n_train_y, n_validation_x, n_validation_y, epoch_count,
                                 patience, batch_size)
    models['MLP'].save(models_path + 'MLP')
else:
    models['MLP'] = keras.models.load_model(models_path + 'MLP')

plot_keras_model(models['MLP'])

forecasts['MLP'] = forecast(models['MLP'], n_test_x, test_y.index, scaler_y)

# -------------------- DEEP CNN --------------------

if do_train:
    print('Training CNN')
    models['CNN'] = build_deep_cnn(features)
    models['CNN'], _ = keras_fit(models['CNN'], n_train_x, n_train_y, n_validation_x, n_validation_y, epoch_count,
                                 patience, batch_size)
    models['CNN'].save(models_path + 'CNN')
else:
    models['CNN'] = keras.models.load_model(models_path + 'CNN')

plot_keras_model(models['CNN'])

forecasts['CNN'] = forecast(models['CNN'], n_test_x, test_y.index, scaler_y)

# -------------------- DEEP RNN --------------------
if do_train:
    print('Training RNN')
    models['RNN'] = build_deep_rnn(features)
    models['RNN'], _ = keras_fit(models['RNN'], n_train_x, n_train_y, n_validation_x, n_validation_y, epoch_count,
                                 patience, batch_size)
    # save model
    models['RNN'].save(models_path + 'RNN')
else:
    models['RNN'] = keras.models.load_model(models_path + 'RNN')

plot_keras_model(models['RNN'])
# evaluate
forecasts['RNN'] = forecast(models['RNN'], n_test_x, test_y.index, scaler_y)

# -------------------- GRU --------------------
if do_train:
    print('Training GRU')
    models['GRU'] = build_deep_gru(features)
    models['GRU'], _ = keras_fit(models['GRU'], n_train_x, n_train_y, n_validation_x, n_validation_y, epoch_count,
                                 patience, batch_size)
    # save model
    models['GRU'].save(models_path + 'GRU')
else:
    models['GRU'] = keras.models.load_model(models_path + 'GRU')

plot_keras_model(models['GRU'])
# evaluate
forecasts['GRU'] = forecast(models['GRU'], n_test_x, test_y.index, scaler_y)

# -------------------- LSTM --------------------
if do_train:
    print('Training LSTM')
    models['LSTM'] = build_lstm(features)
    models['LSTM'], _ = keras_fit(models['LSTM'], n_train_x, n_train_y, n_validation_x, n_validation_y, epoch_count,
                                  patience, batch_size)
    # save model
    models['LSTM'].save(models_path + 'LSTM')
else:
    models['LSTM'] = keras.models.load_model(models_path + 'LSTM')

# evaluate
forecasts['LSTM'] = forecast(models['LSTM'], n_test_x, test_y.index, scaler_y)

# -------------------- COMPARISON --------------------

errors = {algo: RMSE(test_y, forcast) for algo, forcast in forecasts.items()}
algo_by_error = sorted(errors, key=lambda v: errors[v])
best_algo = algo_by_error[0]

print('RMSEs:')
for algo in algo_by_error[::-1]:
    print(f'{algo} -> {errors[algo]}')
    plot_forecast(algo, test_y, forecasts[algo])

# -------------------- DIEBOLD-MARIANO TEST --------------------

dm_matrix = pd.DataFrame({algo_1: [compare_models(test_y, forecasts[algo_1], forecasts[algo_2])
                                   for algo_2 in algo_by_error] for algo_1 in algo_by_error}, index=algo_by_error)
dm_matrix = dm_matrix.iloc[1:, :-1]
mask = np.triu(np.ones_like(dm_matrix, dtype=bool), k=1)

plt.figure(figsize=(15, 4))
sns.heatmap(dm_matrix, mask=mask, cbar=False)
plt.xticks(rotation=0)
plt.show()
