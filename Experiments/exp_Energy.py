# -*- coding: utf-8 -*-
# @Author  : Genglong Li
# @Time    : 2023/7/13 23:50
# @Comment : Compare Clean4MTS with other benchmark times series data cleaning methods on the Energy dataset
from DQDiscovery.LearnCRR.core import DataBase
import numpy as np
import pandas as pd
import pickle
import time
from TSRepair.Profiling import Clean4MTS, Clean4MTSPlus
from TSRepair.Benchmark import Clean4MTSNR
from TSRepair.Benchmark import variance_constraint_clean, speed_constraint_clean_local, speed_constraint_clean_global,\
    speed_plus_acceleration_constraint_clean_global, speed_plus_acceleration_constraint_clean_local, ewma, IMR

from TSRepair.EvaluationMetrics import mnad, rra, precision_recall, mae_error


def performance_evaluation_clean4mts(mode='Clean4MTS'):
    path = r'..\Datasets\Energy_Clean.csv'
    o = open(path)
    df = pd.read_csv(o)
    o.close()
    target = ['T3']
    src = ['T5', 'T1', 'T4']
    d_clean = np.array(df[src + target])

    path = r'..\Datasets\Energy_Dirty.csv'
    o = open(path)
    df = pd.read_csv(o)
    o.close()
    d_dirty = np.array(df[src + target])

    db1 = DataBase(df)
    # The quality constraints discovered before
    with open('Save/rr_EDA.pickle', 'rb') as f:
        rr = pickle.load(f)
        # 测试读取后的Model
    with open('Save/ac_EDA.pickle', 'rb') as f:
        ac = pickle.load(f)
        # 测试读取后的Model
    with open('Save/sc_EDA.pickle', 'rb') as f:
        sc = pickle.load(f)
    st = time.time()
    if mode == 'Clean4MTS+':
        pf = Clean4MTSPlus(db1, rr, sc, ac, 'Timestamp', target=target, src=src, delta_w=12, method='studGA',
                           trappedValue=1e-3, w1=6, w2=2, beta=1, print_repair_message=False)
    elif mode == 'Clean4MTSNR':
        pf = Clean4MTSNR(db1, rr, sc, 'Timestamp', target=target, src=src, delta_w=12, method='studGA',
                         trappedValue=1e-3,
                         w1=6, w2=2, beta=1, print_repair_message=False)
    else:
        pf = Clean4MTS(db1, rr, sc, 'Timestamp', target=target, src=src, delta_w=12, method='studGA', trappedValue=1e-3,
                       w1=6, w2=2, beta=1, print_repair_message=False)
    pf.data_cleaning()
    ed = time.time()
    print(mode)
    d_repair = np.array(pf.dataframe[src + target])
    print('MNAD:', 'after repair', mnad(d_repair=d_repair, d_clean=d_clean), 'before repair',
          mnad(d_repair=d_dirty, d_clean=d_clean))
    print('RRA:', rra(d_repair=d_repair, d_clean=d_clean, d_dirty=d_dirty))
    print('(P, R):', precision_recall(d_repair=d_repair, d_clean=d_clean, d_dirty=d_dirty))
    print('Time Cost:', ed - st)
    print('MAE_ERR:', mae_error(d_repair=d_repair, d_clean=d_clean, d_dirty=d_dirty))


def performance_evaluation_benchmark(mode='Vari'):
    path = r'..\Datasets\Energy_Clean.csv'
    o = open(path)
    df = pd.read_csv(o)
    o.close()
    target = ['T3']
    src = ['T5', 'T1', 'T4']
    d_clean = np.array(df[src + target])

    path = r'..\Datasets\Energy_Dirty.csv'
    o = open(path)
    df = pd.read_csv(o)
    o.close()
    d_dirty = np.array(df[src + target])

    db1 = DataBase(df)
    with open('Save/vc_EDA.pickle', 'rb') as f:
        vc = pickle.load(f)
        # 测试读取后的Model
    with open('Save/ac_EDA.pickle', 'rb') as f:
        ac = pickle.load(f)
        # 测试读取后的Model
    with open('Save/sc_EDA.pickle', 'rb') as f:
        sc = pickle.load(f)

    st = time.time()
    if mode == 'Vari':
        variance_constraint_clean(db1, vc, target + src, 'Timestamp')
    elif mode == 'Speed(L)':
        speed_constraint_clean_local(db1, sc, target + src, 'Timestamp', w=10)
    elif mode == 'Speed(G)':
        speed_constraint_clean_global(db1, sc, target + src, 'Timestamp', w=10)
    elif mode == 'Speed+Acc(G)':
        speed_plus_acceleration_constraint_clean_global(db1, sc, ac, target + src, 'Timestamp', w=10)
    elif mode == 'Speed+Acc(L)':
        speed_plus_acceleration_constraint_clean_local(db1, sc, ac, target + src, 'Timestamp', w=10)
    elif mode == 'EWMA':
        ewma(db1, target + src, sliding=5)
    elif mode == 'IMR':
        path = r'..\Datasets\Energy_Label.csv'
        o = open(path)
        df_label = pd.read_csv(o)
        o.close()
        dblabel = DataBase(df_label)
        delta = {attr: sc[attr][1] / 10 for attr in src + target}
        imr = IMR(db1, dblabel, target + src, p=2, delta=delta, max_iter=10000) # IMR with 20% labels
        imr.repair()
    else:
        print('Please choose the following method:')
        print('Vari/Speed(L)/Speed(G)/Speed+Acc(L)/Speed+Acc(G)/EWMA')
        return
    ed = time.time()
    print(mode)
    d_repair = np.array(db1.dataframe[src + target])
    print('MNAD:', 'after repair', mnad(d_repair=d_repair, d_clean=d_clean), 'before repair',
          mnad(d_repair=d_dirty, d_clean=d_clean))
    print('RRA:', rra(d_repair=d_repair, d_clean=d_clean, d_dirty=d_dirty))
    print('(P, R):', precision_recall(d_repair=d_repair, d_clean=d_clean, d_dirty=d_dirty))
    print('Time Cost:', ed - st)
    print('MAE_ERR:', mae_error(d_repair=d_repair, d_clean=d_clean, d_dirty=d_dirty))


if __name__ == '__main__':
    performance_evaluation_clean4mts()
    for m in ['Vari', 'Speed(L)', 'Speed(G)', 'Speed+Acc(L)', 'Speed+Acc(G)', 'EWMA', 'IMR']:
        performance_evaluation_benchmark(mode=m)