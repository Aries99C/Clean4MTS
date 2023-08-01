# -*- coding:utf-8 -*-
# @Author : Genglong Li
# @Time : 2023/3/30 20:11
# @Comment: Learning quality constraints in the form of regression rules

import pandas as pd
import copy
import random
import time
import numpy as np
from DQDiscovery.LearnCRR.lineartree2 import LinearTreeRegressor, LinearBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor


reg_models = {'bayesian': BayesianRidge, 'kernel': KernelRidge, 'linear': LinearRegression, 'pls': PLSRegression,
              'elastic': ElasticNet, 'linearboost': LinearBoostRegressor}
'''
reg_models = {'bayesian': BayesianRidge, 'robust': RANSACRegressor, 'kernel': KernelRidge,
              'linear': LinearRegression, 'mlp': MLPRegressor}
'''


class DataBase:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe


class Range:
    def build(self, db: DataBase, partition_attr: list):
        # source_attr: attributes in filtering
        self.l = dict(db.dataframe[partition_attr].min())
        self.r = dict(db.dataframe[partition_attr].max())
        self.attr = copy.deepcopy(partition_attr)

    def __init__(self, l: dict, r: dict):
        self.l = copy.copy(l)
        self.r = copy.copy(r)
        self.inv = set()
        self.attr = list(l.keys())

    def binary(self, attr): #
        lx, rx = copy.copy(self.l), copy.copy(self.r)
        lx[attr] = (self.l[attr] + self.r[attr]) / 2.0
        ly, ry = copy.copy(self.l), copy.copy(self.r)
        ry[attr] = (self.l[attr] + self.r[attr]) / 2.0
        return [Range(lx, rx), Range(ly, ry)]

    def triary(self, attr):
        lx, rx = copy.deepcopy(self.l), copy.deepcopy(self.r)
        ly, ry = copy.deepcopy(self.l), copy.deepcopy(self.r)
        lz, rz = copy.deepcopy(self.l), copy.deepcopy(self.r)
        rx[attr] = (2 * self.l[attr] + self.r[attr]) / 3.0
        ly[attr], ry[attr] = (2 * self.l[attr] + self.r[attr]) / 3.0, (self.l[attr] + 2 * self.r[attr]) / 3.0
        lz[attr] = (self.l[attr] + 2 * self.r[attr]) / 3.0
        return [Range(lx, rx), Range(ly, ry), Range(lz, rz)]

    def limit_dom(self, attr, lm, rm):
        self.l[attr], self.r[attr] = lm, rm
        return self

    def clone(self):
        return Range(self.l, self.r)


def partition(rangex: Range, part_meth, attr, adaptive_val=None):  # partition the Range on one attribute
    assert attr in rangex.attr
    if part_meth == 'binary':
        return rangex.binary(attr)
    elif part_meth == 'triary':
        return rangex.triary(attr)
    elif part_meth == 'median':
        return [rangex.clone().limit_dom(attr, rangex.l[attr], adaptive_val),
                rangex.clone().limit_dom(attr, adaptive_val, rangex.r[attr])]
    else:
        return [None]


def train_val_split(db: DataBase, src: list, target: list, rate=0.2, granularity=10):
    assert len(target) == 1
    assert 0 < rate < 0.5
    id_num = db.dataframe.shape[0]
    train_id, val_id = [], []
    e_id = 0
    while e_id < id_num - granularity:  # assume the indexes are in time-order
        id_set = {x for x in range(granularity)}
        val_set = set(random.sample(id_set, int(rate*granularity)))
        train_set = id_set - val_set
        train_list = [x + e_id for x in train_set]
        val_list = [x + e_id for x in val_set]
        train_id += train_list
        val_id += val_list
        e_id += granularity
    train_id += [x for x in range(e_id, id_num)]
    train_X, train_Y, val_X, val_Y = db.dataframe[src].iloc[train_id], db.dataframe[target].iloc[
        train_id], db.dataframe[src].iloc[val_id], db.dataframe[target].iloc[val_id]
    return train_X, train_Y, val_X, val_Y, train_id, val_id  # return pd.DataFrame (2-dimensions) and indexes


def train_val_split_array(X: np.ndarray, Y: np.ndarray, rate=0.2, granularity=10):
    assert 0 < rate < 0.5
    assert X.shape[0] == Y.shape[0]
    id_num = X.shape[0]
    train_id, val_id = [], []
    e_id = 0
    while e_id < id_num - granularity:  # assume the indexes are in time-order
        id_set = {x for x in range(granularity)}
        val_set = set(random.sample(id_set, int(rate*granularity)))
        train_set = id_set - val_set
        train_list = [x + e_id for x in train_set]
        val_list = [x + e_id for x in val_set]
        train_id += train_list
        val_id += val_list
        e_id += granularity
    train_id += [x for x in range(e_id, id_num)]
    train_X, train_Y, val_X, val_Y = X[train_id, :], Y[train_id],  X[val_id, :], Y[val_id]
    return train_X, train_Y, val_X, val_Y  # return np.ndarray and indexes


def val_model(models: list, valX: np.ndarray, valY: np.ndarray, eps: float):  # the element of 'model': 2-tuple(the name of the model, the regression model)
    best_e, error, md, deltay = np.inf, np.inf, None, -1
    for i in range(len(models)):
        mdl = models[i][0]
        model = models[i][1]
        if mdl == 'bayesian':
            bias = np.array(valY - model.predict(np.vander(valX.T[0], N=4)))
        else:
            bias = np.array(valY - model.predict(valX))
        e = (np.max(bias) - np.min(bias)) / 2.0
        if e <= eps:
            avg = (np.max(bias) + np.min(bias)) / 2.0
            err = np.sum(np.abs(bias - avg)) / (np.shape(valY)[0] * 1.0)
            if err < error:
                best_e, error, md, deltay = e, err, models[i], (np.max(bias) + np.min(bias)) / 2.0
    return best_e, error, md, deltay


def filter_sv(X: np.array, Y: np.array, P: np.ndarray, rangex: Range, partition_attr):
    assert partition_attr in rangex.attr
    ans = []
    for p in P:
        if rangex.l[partition_attr] <= p <= rangex.r[partition_attr]:
            ans.append(True)
        else:
            ans.append(False)
    return X[np.array(ans)], Y[np.array(ans)], P[np.array(ans)]


'''
    for x in X:
        flag = True
        for i in range(len(rangex.attr)):
            attr = rangex.attr[i]
            if rangex.l[attr] <= x[i] <= rangex.r[attr]:
                continue
            else:
                flag = False
                break
        ans.append(flag)
    return X[np.array(ans)], Y[np.array(ans)]

'''


def median(subX):
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    a, b, c = subX[0], subX[int(len(subX) / 2)], subX[-1]
    if a < b:
        if b < c: return b
        if a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


def get_pipline(regmdl, **kw):
    if regmdl == 'linear':
        estimators = [('standardize', RobustScaler()), ('reg', LinearRegression())]
        return Pipeline(estimators)
    elif regmdl == 'bayesian':
        reg = BayesianRidge(tol=random.random() * 1e-5, fit_intercept=False, compute_score=True)
        init = [1 / np.var(kw['train_y']), 1.]
        reg.set_params(alpha_init=init[0], lambda_init=init[1])
        return reg
    elif regmdl == 'kernel':
        estimators = [('standardize', RobustScaler()), ('reg', KernelRidge(alpha=1.0, kernel='rbf', gamma=0.04))]
        return Pipeline(estimators)
    elif regmdl == 'pls':
        estimators = [('standardize', RobustScaler()), ('reg', PLSRegression())]
        return Pipeline(estimators)
    elif regmdl == 'elastic':
        estimators = [('standardize', RobustScaler()), ('reg', ElasticNet(random_state=0))]
        return Pipeline(estimators)
    elif regmdl == 'linearboost':
        reg = LinearRegression()
        estimators = [('standardize', RobustScaler()), ('reg', LinearBoostRegressor(reg))]
        return Pipeline(estimators)
    else:
        raise TypeError('The inputed regression model is not considered by the system')


def alg1_discover(db: DataBase, target: list, src: list, partition_range: Range, partition_attr: str, regmds=None,
                      part_meth='binary', eps=0.1, edge_sz=150, **kw):
    assert partition_attr in db.dataframe.columns
    assert partition_attr not in target
    assert partition_attr in partition_range.attr
    if regmds is None:
        global reg_models
        regmds = reg_models
    else:
        assert type(regmds) is dict
    l = db.dataframe.shape[0]
    x, y = np.array(db.dataframe[src]), np.array(db.dataframe[target]).reshape([l, ])
    p = np.array(db.dataframe[partition_attr])
    que = [partition_range]
    model = []
    crr = []

    while len(que) > 0:
        error, ext_md, deltay = np.inf, None, -1.0
        rmse = np.inf
        r0 = que.pop(0)
        sub_x, sub_y, sub_p = filter_sv(x, y, p, r0, partition_attr)
        if sub_x.shape[0] == 0:
            continue
        if len(model) > 0:
            error, rmse, ext_md, deltay = val_model(models=model, valX=sub_x, valY=sub_y, eps=eps)
        if rmse <= eps:     # 考虑怎么定义rho，是否需要包容潜在的outlier（是：error, 否：rmse * 3)
            for i in range(len(model)):
                if model[i][1] == ext_md:
                    crr[i][2].append((r0, deltay))
                    if rmse * 3 > crr[i][1]:
                        crr[i][1] = rmse * 3
                    break
        else:
            train_x, train_y, val_x, val_y = train_val_split_array(sub_x, sub_y)
            best_reg, bestloss, bestmdl = None, None, None
            for regmdl in regmds.keys():
                if regmdl != 'bayesian':
                    if sub_x.shape[0] > 1000 and regmdl == 'kernel':
                        continue
                    reg = get_pipline(regmdl)
                    reg.fit(train_x, train_y)
                    loss = np.sum(np.abs(val_y - reg.predict(val_x))) / (np.shape(val_x)[0] * 1.0)
                else:
                    reg = get_pipline(regmdl, train_y=train_y)
                    reg.fit(np.vander(train_x.T[0], N=4), train_y)
                    loss = np.sum(np.abs(val_y - reg.predict(np.vander(val_x.T[0], N=4)))) / (np.shape(val_x)[0] * 1.0)
                if bestloss is None or loss < bestloss:
                    best_reg = reg
                    bestloss = loss
                    bestmdl = regmdl
            if bestmdl != 'bayesian':
                maxerror = np.max(np.abs(val_y - best_reg.predict(val_x)))
                rmse = np.sqrt(mean_squared_error(best_reg.predict(val_x), val_y))
            else:
                maxerror = np.max(np.abs(val_y - best_reg.predict(np.vander(val_x.T[0], N=4))))
                rmse = np.sqrt(mean_squared_error(best_reg.predict(np.vander(val_x.T[0], N=4)), val_y))
            if sub_x.shape[0] <= edge_sz or rmse <= eps:  # 考虑怎么定义rho，是否需要包容潜在的outlier（是：maxerror, 否：rmse * 3)
                model.append((bestmdl, best_reg))  # 2-tuple(the name of the model, the regression model)
                # r0 = r0.clone().limit_dom(partition_attr, np.min(sub_p), np.max(sub_p))
                crr.append([(bestmdl, best_reg), rmse * 3, [(r0, 0.0)]])
                print(sub_x.shape[0])

            else:
                rs = partition(r0, part_meth=part_meth, attr=partition_attr)
                if rs is not None:
                    que += rs

    return crr, model


def alg1_discover_rel(db: DataBase, src: list, target: list, depth=3):
    # linear
    # reg = LinearRegression()
    # ridge
    reg = ElasticNet(random_state=0)
    # mlp
    # reg = MLPRegressor(random_state=1)
    ltr = LinearBoostRegressor(reg, max_depth=depth)
    trainX, trainY, valX, valY, train_id, test_id = train_val_split(db, src, target)

    ltr.fit(trainX, trainY)
    rmse = np.sqrt(mean_squared_error(ltr.predict(valX), valY))

    '''
    crr = ltr.alg2_compation(valX)
    st = time.time()
    pred, mask = np.zeros(valX.shape[0]), np.zeros(valX.shape[0])
    for id in crr.keys():
        for msk, bias in crr[id]:
            pred[msk] = ltr._ext_models[id].predict(valX[msk]) + bias
            mask += msk
    mask = mask > 0
    if mask.sum() > 0:
        rmse = np.sqrt(mean_squared_error(pred[mask], valY[mask]))
    else:
        rmse = 0.
    rmse /= (np.sum(np.abs(valY[mask])) / (1.0 * np.sum(mask)))
    # print(time.time() - st, rmse, len(crr))
    '''

    return [('lineartree', ltr), rmse * 3, [(None, 0.0)]]


if __name__ == '__main__':
    # path = r'E:\2022下半年\时序数据质量管理\experiment_1\experiment_1.csv'
    path = r'E:\2021秋\工业大数据\数据集\Gecco industrial challenge 2018 dataset\1_gecco2018_water_quality.csv'
    o = open(path)
    df = pd.read_csv(o)
    o.close()
    db1 = DataBase(df.iloc[0:15000])
    r1 = Range({}, {})
    r1.build(db1, partition_attr=['index'])
    target = ['pH']
    src = ['Tp', 'Cl', 'Redox', 'Leit',
                     'Cl_2']
    crr1, model1 = alg1_discover(db1, target, src, r1, 'index',  eps=0.02, edge_sz=100)
    print(model1)
    # crr = alg1_discover_rel(db1, src, target, 10)
    # print('hh', crr[1])



