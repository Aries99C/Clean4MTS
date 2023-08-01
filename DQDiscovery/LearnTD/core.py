# -*- coding:utf-8 -*-
# @Author : Genglong Li
# @Time : 2023/4/1 15:14
# @Comment: Learning quality constraints for temporal dependencies
from DQDiscovery.LearnCRR.core import Range
from DQDiscovery.LearnCRR.core import DataBase, alg1_discover, alg1_discover_rel
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pickle


def density_plot(arr: np.ndarray):
    plt.hist(pd.Series(arr), density=True, edgecolor='w', bins=50)
    plt.show()


class SpeedConstraint:
    def __init__(self, attr):
        self.attr = attr
        self.hasdcv = False
        self.v_lb = None
        self.v_ub = None
        self.range = None
        self.v = None

    def set(self, lb, ub):
        assert lb < ub
        self.v_lb = lb
        self.v_ub = ub

    def discover(self, db: DataBase,  t_attr, rangex=None, sample_num=500, confidence=0.99, keepv=False):
        assert self.attr in db.dataframe.columns
        assert t_attr in db.dataframe.columns  # Temporal attribute (Preprocessed)(e.g. timestamps)
        cond = None
        if rangex is not None:
            assert isinstance(rangex, Range)
            for attr in rangex.attr:
                assert attr in db.dataframe.columns
                if cond is None:
                    cond = (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
                else:
                    cond = cond & (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            timeseries = np.array(db.dataframe[cond][[t_attr, self.attr]])
            self.range = rangex
        else:
            timeseries = np.array(db.dataframe[[t_attr, self.attr]])
        v = []
        if timeseries.shape[0]-1 > sample_num:
            sind = np.random.choice(a=timeseries.shape[0]-1, size=sample_num, replace=False, p=None)
        else:
            sind = [x for x in range(timeseries.shape[0]-1)]
        for i in sind:
            v.append((timeseries[i + 1, 1] - timeseries[i, 1]) / (timeseries[i + 1, 0] - timeseries[i, 0]))
        v = np.array(v)
        self.v_lb, self.v_ub = st.t.interval(confidence=confidence, loc=0, df=1, scale=st.sem(v))
        self.hasdcv = True
        print(self.attr, self.v_lb, self.v_ub)    # ***
        density_plot(v)    # ***
        if keepv is True:
            self.v = v


class AccelerationConstraint:
    def __init__(self, attr):
        self.attr = attr
        self.hasdcv = False
        self.a_lb = None
        self.a_ub = None
        self.range = None
        self.a = None

    def set(self, lb, ub):
        assert lb < ub
        self.a_lb = lb
        self.a_ub = ub

    def discover(self, db: DataBase,  t_attr, rangex=None, sample_num=500, confidence=0.99, keepa=False):
        assert self.attr in db.dataframe.columns
        assert t_attr in db.dataframe.columns  # Timestamp attribute (Preprocessed)
        cond = None
        if rangex is not None:
            assert isinstance(rangex, Range)
            for attr in rangex.attr:
                assert attr in db.dataframe.columns
                if cond is None:
                    cond = (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
                else:
                    cond = cond & (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            timeseries = np.array(db.dataframe[cond][[t_attr, self.attr]])
            self.range = rangex
        else:
            timeseries = np.array(db.dataframe[[t_attr, self.attr]])
        a = []
        if timeseries.shape[0]-2 > sample_num:
            sind = np.random.choice(a=timeseries.shape[0]-2, size=sample_num, replace=False, p=None)
        else:
            sind = [x for x in range(timeseries.shape[0]-2)]
        for i in sind:
            v_i1 = (timeseries[i + 1, 1] - timeseries[i, 1]) / (timeseries[i + 1, 0] - timeseries[i, 0])
            v_i2 = (timeseries[i + 2, 1] - timeseries[i + 1, 1]) / (timeseries[i + 2, 0] - timeseries[i + 1, 0])
            a.append((v_i2 - v_i1) / (timeseries[i + 2, 0] - timeseries[i + 1, 0]))
        a = np.array(a)
        self.a_lb, self.a_ub = st.t.interval(confidence=confidence, loc=0, df=1, scale=st.sem(a))
        self.hasdcv = True
        print(self.attr, self.a_lb, self.a_ub)    # ***
        density_plot(a)    # ***
        if keepa is True:
            self.a = a


class VarianceConstraint:
    def __init__(self, attr):
        self.attr = attr
        self.range = None
        self.hasdcv = False
        self.var_b = None
        self.var = None

    def discover(self, db: DataBase,  t_attr, w, rangex=None, sample_num=500, confidence=0.99, keepvar=False):
        assert self.attr in db.dataframe.columns
        assert t_attr in db.dataframe.columns  # Timestamp attribute (Preprocessed)
        cond = None
        if rangex is not None:
            assert isinstance(rangex, Range)
            for attr in rangex.attr:
                assert attr in db.dataframe.columns
                if cond is None:
                    cond = (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
                else:
                    cond = cond & (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            timeseries = np.array(db.dataframe[cond][[t_attr, self.attr]])
            self.range = rangex
        else:
            timeseries = np.array(db.dataframe[[t_attr, self.attr]])
        var = []
        if timeseries.shape[0] - w > sample_num:
            sind = np.random.choice(a=timeseries.shape[0] - w, size=sample_num, replace=False, p=None)
        else:
            sind = [x for x in range(timeseries.shape[0] - w)]
        for i in sind:
            var.append(np.var(timeseries[i: i+w, 1]))
        var = np.array(var)
        self.var_b = np.quantile(var, confidence)# (w-1)*np.mean(var)/st.chi2.ppf(1-confidence, w-1)
        self.hasdcv = True
        print(self.attr, self.var_b)    # ***
        density_plot(var)    # ***
        if keepvar is True:
            self.var = var


def discover_speed_constraints(db: DataBase, attrl: list,  t_attr, rangex=None,
                               sample_num=500, confidence=0.99, keepv=False):
    speed_constraints = {}
    cond = None

    if rangex is not None:
        assert isinstance(rangex, Range)
        for attr in rangex.attr:
            assert attr in db.dataframe.columns
            if cond is None:
                cond = (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            else:
                cond = cond & (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            db = DataBase(db.dataframe[cond])

    for attr in attrl:
        assert attr in db.dataframe.columns
        sc = SpeedConstraint(attr=attr)
        sc.discover(db=db, t_attr=t_attr, sample_num=sample_num, confidence=confidence, keepv=keepv)
        speed_constraints[attr] = (sc.v_lb, sc.v_ub)
    return speed_constraints


def discover_acceleration_constraints(db: DataBase, attrl: list,  t_attr, rangex=None,
                               sample_num=500, confidence=0.98, keepa=False):
    acceleration_constraints = {}
    cond = None

    if rangex is not None:
        assert isinstance(rangex, Range)
        for attr in rangex.attr:
            assert attr in db.dataframe.columns
            if cond is None:
                cond = (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            else:
                cond = cond & (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            db = DataBase(db.dataframe[cond])

    for attr in attrl:
        assert attr in db.dataframe.columns
        ac = AccelerationConstraint(attr=attr)
        ac.discover(db=db, t_attr=t_attr, sample_num=sample_num, confidence=confidence, keepa=keepa)
        acceleration_constraints[attr] = (ac.a_lb, ac.a_ub)
    return acceleration_constraints


def discover_variance_constraints(db: DataBase, attrl: list,  t_attr, w, rangex=None,
                               sample_num=500, confidence=0.99, keepvar=False):
    variance_constraints = {}
    cond = None

    if rangex is not None:
        assert isinstance(rangex, Range)
        for attr in rangex.attr:
            assert attr in db.dataframe.columns
            if cond is None:
                cond = (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            else:
                cond = cond & (db.dataframe[attr] >= rangex.l[attr]) & (db.dataframe[attr] < rangex.r[attr])
            db = DataBase(db.dataframe[cond])

    for attr in attrl:
        assert attr in db.dataframe.columns
        vc = VarianceConstraint(attr=attr)
        vc.discover(db=db, t_attr=t_attr, w=w, sample_num=sample_num, confidence=confidence, keepvar=keepvar)
        variance_constraints[attr] = vc.var_b
    return variance_constraints
