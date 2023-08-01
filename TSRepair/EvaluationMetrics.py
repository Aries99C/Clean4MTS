# -*- coding:utf-8 -*-
# @Author : Genglong Li
# @Time : 2023/5/21 11:52
# @Comment: Evaluation Metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DQDiscovery.LearnCRR.core import DataBase
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def mse(d_repair: np.ndarray, d_clean: np.ndarray):
    # Assume that the input of the scaler is a DataFrame
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    # data_dirty_scaled = scaler.transform(db_dirty.dataframe[ra])
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        # denominator = mean_squared_error(data_clean_scaled[:, i], data_dirty_scaled[:, i])
        # if denominator == 0:
            # continue
        sum += mean_squared_error(data_clean_scaled[:, i], data_repair_scaled[:, i]) # / denominator
    return sum / dim


def mnad(d_repair: np.ndarray, d_clean: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    # data_dirty_scaled = scaler.transform(db_dirty.dataframe[ra])
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        # denominator = mean_absolute_error(data_clean_scaled[:, i], data_dirty_scaled[:, i])
        # if denominator == 0:
            # continue
        sum += mean_absolute_error(data_clean_scaled[:, i], data_repair_scaled[:, i]) # / denominator
    return sum / dim


def rra(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    # Assume that the input of the scaler is a DataFrame
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    data_dirty_scaled = scaler.transform(d_dirty)
    data_repair_scaled = scaler.transform(d_repair)
    mae_rc = mnad(d_repair, d_clean)
    mae_rd = 0
    mae_cd = mnad(d_dirty, d_clean)
    dim = data_repair_scaled.shape[1]
    for i in range(dim):
        mae_rd += mean_absolute_error(data_dirty_scaled[:, i], data_repair_scaled[:, i])
    mae_rd /= dim
    return 1 - mae_rc / (mae_cd + mae_rd)


def precision_recall(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    bound = (np.max(d_dirty, axis=0) - np.min(d_dirty, axis=0)) / 10000
    error_array = np.abs(d_dirty - d_clean) > bound
    repair_array = np.abs(d_dirty - d_repair) > bound
    benefit_repair_array = (np.abs(d_clean - d_repair) < np.abs(d_dirty - d_clean)) * repair_array
    tp = error_array * benefit_repair_array
    # print(np.sum(tp != 0), np.sum(repair_array != 0))
    return np.sum(tp != 0) / (np.sum(repair_array != 0)), np.sum(tp != 0) / (np.sum(error_array != 0))


def mae_error(d_repair: np.ndarray, d_clean: np.ndarray, d_dirty: np.ndarray):
    scaler = MinMaxScaler()
    scaler.fit(d_clean)
    d_repair = np.where(d_dirty == d_clean, d_clean, d_repair)
    # data_dirty_scaled = scaler.transform(db_dirty.dataframe[ra])
    data_repair_scaled = scaler.transform(d_repair)
    data_clean_scaled = scaler.transform(d_clean)
    sum = 0
    dim = data_clean_scaled.shape[1]
    for i in range(dim):
        sum += mean_absolute_error(data_clean_scaled[:, i], data_repair_scaled[:, i]) # / denominator
    return sum / dim