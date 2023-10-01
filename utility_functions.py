# PREPROCESSING UTILITY FUNCTIONS

# used for cycling data
import os

import numpy as np
import pandas as pd
import pickle
# sklearn training
from pandas import Series

# statsmodels
from statsmodels.tsa.stattools import adfuller, kpss
# keras
from keras.utils.vis_utils import plot_model
# metrics
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# graphics
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

from dm_test import dm_test

rcParams['figure.figsize'] = (10, 6)


def cycles(value, n_values):
    argument = 2 * np.pi * value / n_values
    return np.sin(argument), np.cos(argument)


def adf_kpss_results(timeseries, max_d):
    results = []

    for idx in range(max_d):
        adf_result = adfuller(timeseries, autolag='AIC')
        kpss_result = kpss(timeseries, regression='c', nlags="auto")
        timeseries = timeseries.diff().dropna()
        if adf_result[1] <= 0.05:
            adf_stationary = True
        else:
            adf_stationary = False
        if kpss_result[1] <= 0.05:
            kpss_stationary = False
        else:
            kpss_stationary = True

        stationary = adf_stationary & kpss_stationary

        results.append((idx, adf_result[1], kpss_result[1], adf_stationary, kpss_stationary, stationary))

    # Construct DataFrame
    results_df = pd.DataFrame(results, columns=['d', 'adf_stats', 'p-value', 'is_adf_stationary', 'is_kpss_stationary',
                                                'is_stationary'])

    return results_df


# TRAIN UTILITY FUNCTIONS

def to_series(forecasts_values, indexes):
    return Series(forecasts_values, index=indexes)


def take_last(*datasets, quantity=1, what='day'):
    splits = []
    for dataset in datasets:

        if what is 'day':
            cut = 24 * quantity
        elif what is 'week':
            cut = 24 * 7 * quantity
        elif what is 'month':
            cut = 24 * 31 * quantity
        else:
            cut = 24 * 365 * quantity

        splits.append(dataset[:-cut].copy())
        splits.append(dataset[-cut:].copy())

    return tuple(splits)


def add_one_dim(data):
    return data.reshape(data.shape[0], data.shape[1], 1)


def save_model(model, path):
    with open(path, 'wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def split_continuously(*datas, partitions):
    partitions = np.atleast_1d(partitions)
    split_data = []
    for data in datas:
        initial_size = len(data)
        for p in partitions:
            size = int(initial_size * p)
            split_data.append(data[:size])
            data = data[size:]
    return tuple(split_data)


def plots(data, labels, title=None):
    for d in data:
        plt.plot(d)
    plt.legend(labels)
    if title is not None:
        plt.title(title)
    plt.show()


def normalize_2d(fit_data, *transform_data, transform_function):
    scaler = transform_function.fit(fit_data)
    return (scaler, *map(lambda v: scaler.transform(v), transform_data))


def normalize_1d(fit_data, *transform_data, transform_function):
    scaler = transform_function.fit(fit_data.values.reshape(-1, 1))
    return (scaler, *map(lambda v: scaler.transform(v.values.reshape(-1, 1)).squeeze(), transform_data))


def forecast(model, test_x, indexes, scaler):
    normalized_forecasts = model.predict(test_x)
    # reversed forecasts
    return to_series(scaler.inverse_transform(normalized_forecasts.reshape(-1, 1)).flatten(), indexes)


def endog_forecast(model, test_y, scaler_y, exog=None):
    n_forecasts = model.forecast(steps=len(test_y), exog=exog) if exog is not None else model.forecast(
        steps=len(test_y))
    forecasts = scaler_y.inverse_transform(n_forecasts.reshape(-1, 1)).flatten()
    return to_series(forecasts, test_y.index)


def RMSE(real, forecasted_values):
    return mean_squared_error(real, forecasted_values, squared=False)


def compare_models(test_y, f_algo_1, f_algo_2, alpha=0.05):
    return False if dm_test(test_y, f_algo_1, f_algo_2).p_value < alpha / 2 else True


def plot_forecast(model_name, real, forecasted_values):
    # real forecasted plot
    pearson = pearsonr(real, forecasted_values)[0]
    rmse = RMSE(real, forecasted_values)
    plots([real, forecasted_values], ['real', 'forecasted'], f'{model_name}\nRMSE: {rmse}\nPearson: {pearson}')
