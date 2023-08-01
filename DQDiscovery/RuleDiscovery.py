# -*- coding:utf-8 -*-
# @Author : Genglong Li
# @Time : 2023/4/3 10:01
from DQDiscovery.LearnCRR.core import Range, DataBase, alg1_discover
from DQDiscovery.LearnTD.core import discover_speed_constraints
import pandas as pd
import matplotlib.pyplot as plt


def discover_crrs_and_scs(db: DataBase, target: list, src: list, partition_attr: str, tp_attr,
                          edge_sz, eps, regmds=None, part_meth='binary',  sample_num=500, confidence=0.99, keepv=False):
    r1 = Range({}, {})
    r1.build(db, partition_attr=[partition_attr])
    crrs, models = alg1_discover(db, target, src, r1, partition_attr, eps=eps, edge_sz=edge_sz,
                                 regmds=regmds, part_meth=part_meth)
    quality_rules = []
    for crr in crrs:
        condition = None
        for item in crr[2]:  # [(modellabel, regmodel), rmse * 3, [(range1, deltay_1),...]]
            tmpcond = None
            for attr in item[0].attr:
                assert attr in db.dataframe.columns
                if tmpcond is None:
                    tmpcond = (db.dataframe[attr] >= item[0].l[attr]) & (db.dataframe[attr] < item[0].r[attr])
                else:
                    tmpcond = tmpcond & (db.dataframe[attr] >= item[0].l[attr]) & (db.dataframe[attr] < item[0].r[attr])
            if condition is None:
                condition = (tmpcond)
            else:
                condition = condition | (tmpcond)
        print('Regression Rule', crr[0][0], crr[1])
        db = DataBase(db.dataframe[condition])
        print('Speed Constraints')
        scs = discover_speed_constraints(db, src+target, tp_attr, sample_num=sample_num,
                                         confidence=confidence, keepv=keepv)
        # Assume that for each segment of data partitioned by conditions, speed constraints need to discover again.
        quality_rules.append(crr + [scs] + [target] + [src])
    return quality_rules


'''
if __name__ == '__main__':
    path = r'E:\2021秋\工业大数据\数据集\整理完成的引风机数据\MELODY_test\data\fan_B_sample_1.csv'
    # path = r'E:\2021秋\工业大数据\数据集\整理完成的引风机数据\MELODY_test\data\fan_sample_3.csv'
    # path = r'E:\2021秋\工业大数据\数据集\整理完成的引风机数据\整理完成的引风机数据\整理完成的引风机数据\3#机组引风机数据B.csv'
    o = open(path)
    df = pd.read_csv(o)
    o.close()
    db1 = DataBase(df)
    target = ['U3_HNV10CT111']
    src = ['U3_HNV10CT102', 'U3_HNV10CT103', 'U3_HNV10CT104']
    # qr_crr_scs = discover_crrs_and_scs(db1, target, src, 'time', 'time', 2000, 0.3)   # 注意：测试的时候未分块
    df[target+src].iloc[0:1000].plot(subplots=True)
    r_1 = [(420, 440), ['U3_HNV10CT111'], 0.35]
    r_2 = [(460, 480), ['U3_HNV10CT104'], -0.5]
    r_3 = [(520, 540), ['U3_HNV10CT103'], -0.5]
    r_4 = [(575, 595), ['U3_HNV10CT102'], 0.5]
    plt.show()
'''




