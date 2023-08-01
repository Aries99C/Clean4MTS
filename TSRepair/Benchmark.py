# -*- coding: utf-8 -*-
# @Author  : Genglong Li
# @Time    : 2023/6/15 15:53
# @Comment : Benchmark Data Cleaning Algorithms for Time-Series Data
import numpy as np
from scipy.optimize import linprog
from DQDiscovery.LearnCRR.core import DataBase
from statsmodels.tsa.ar_model import AutoReg
from TSRepair.Profiling import Clean4MTS


def solve_quad(A, B, C):
    delta = B ** 2 - 4 * A * C  # 计算delta
    if delta <= 0:  # 判断delta如果小于0
        x = B / (-2 * A)  # 则解为B / (-2*A)
        return x, x
    else:  # 除以上两种情况以外，即delta如果大于0
        x1 = (B + delta ** 0.5) / (-2 * A)  # 计算解1：x1
        x2 = (B - delta ** 0.5) / (-2 * A)  # 计算解2：x2
        return x1, x2


# Speed constraints based data cleaning (Local)
def speed_constraint_clean_local(db: DataBase, speed_constraints, ra: list, t_attr, w=100):
    assert t_attr in db.dataframe.columns
    for attr in ra:
        assert attr in db.dataframe.columns
        assert attr in speed_constraints.keys()
    ra_range = [speed_constraints[attr] for attr in ra]
    ra_vlb = np.array([x[0] for x in ra_range])
    ra_vub = np.array([x[1] for x in ra_range])
    data = np.array(db.dataframe[ra])
    t = np.array(db.dataframe[t_attr])
    for i in range(1, db.dataframe.shape[0]-1):
        x_i_min = ra_vlb * (t[i] - t[i - 1]) + data[i - 1]
        x_i_max = ra_vub * (t[i] - t[i - 1]) + data[i - 1]
        candidate_i = [data[i]]
        for k in range(i + 1, db.dataframe.shape[0]):
            if t[k] > t[i] + w:
                break
            candidate_i.append(ra_vlb * (t[i] - t[k]) + data[k])
            candidate_i.append(ra_vub * (t[i] - t[k]) + data[k])
        candidate_i = np.array(candidate_i)
        x_i_mid = np.median(candidate_i, axis=0)
        x_i_mid = np.where(x_i_mid > x_i_max, x_i_max, x_i_mid)
        x_i_mid = np.where(x_i_mid < x_i_min, x_i_min, x_i_mid)
        data[i] = x_i_mid
    db.dataframe[ra] = data


# Speed constraints based data cleaning (Global)
def speed_constraint_clean_global(db: DataBase, speed_constraints, ra: list, t_attr, w=100,
                                  size=200, overlapping_ratio=0.2):
    assert t_attr in db.dataframe.columns
    for attr in ra:
        assert attr in db.dataframe.columns
        assert attr in speed_constraints.keys()
    ra_range = [speed_constraints[attr] for attr in ra]
    ra_vlb = np.array([x[0] for x in ra_range])
    ra_vub = np.array([x[1] for x in ra_range])
    for k in range(len(ra)):
        s = 0
        while s + size <= db.dataframe.shape[0]:
            data = np.array(db.dataframe.loc[s:min(s + size, db.dataframe.shape[0]), ra])
            t = np.array(db.dataframe.loc[s:min(s + size, db.dataframe.shape[0]), t_attr])
            c = np.ones(2 * data.shape[0])
            A = []
            b = []

            bounds = [(0, None) for j in range(2 * data.shape[0])]
            for i in range(data.shape[0]):
                for j in range(i + 1, data.shape[0]):
                    if t[j] > t[i] + w:
                        break
                    bij_max = ra_vub[k] * (t[j] - t[i]) - (data[j, k] - data[i, k])
                    bij_min = -ra_vlb[k] * (t[j] - t[i]) + (data[j, k] - data[i, k])
                    b.append(bij_max)
                    b.append(bij_min)
                    aij_max = np.zeros(2 * data.shape[0])
                    aij_min = np.zeros(2 * data.shape[0])
                    aij_max[j], aij_max[i] = 1, -1
                    aij_max[j + data.shape[0]], aij_max[i + data.shape[0]] = -1, 1
                    A.append(aij_max)
                    aij_min[j], aij_min[i] = -1, 1
                    aij_min[j + data.shape[0]], aij_min[i + data.shape[0]] = 1, -1
                    A.append(aij_min)
            A = np.array(A)
            b = np.array(b)
            # print(c, A, b)
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            db.dataframe.loc[s:min(s + size, db.dataframe.shape[0]), ra[k]] = (res.x[:data.shape[0]] -
                                                                           res.x[data.shape[0]:]) + data[:, k]
            s += int((1 - overlapping_ratio) * size)


# Speed and acceleration constraints based data cleaning (Global)
def speed_plus_acceleration_constraint_clean_global(db: DataBase, speed_constraints, acceleration_constraints,
                                             ra: list, t_attr, w=50, size=100, overlapping_ratio=0.1):
    assert t_attr in db.dataframe.columns
    for attr in ra:
        assert attr in db.dataframe.columns
        assert attr in speed_constraints.keys()
        assert attr in acceleration_constraints.keys()
    ra_range_v = [speed_constraints[attr] for attr in ra]
    ra_range_a = [acceleration_constraints[attr] for attr in ra]
    ra_vlb = np.array([x[0] for x in ra_range_v])
    ra_vub = np.array([x[1] for x in ra_range_v])
    ra_alb = np.array([x[0] for x in ra_range_a])
    ra_aub = np.array([x[1] for x in ra_range_a])

    for k in range(len(ra)):
        s = 0
        while s + size <= db.dataframe.shape[0]:
            data = np.array(db.dataframe.loc[s:min(s + size, db.dataframe.shape[0]), ra])
            t = np.array(db.dataframe.loc[s:min(s + size, db.dataframe.shape[0]), t_attr])
            c = np.ones(2 * data.shape[0])
            A = []
            b = []

            bounds = [(0, None) for j in range(2 * data.shape[0])]
            for i in range(data.shape[0]):
                for j in range(i + 1, data.shape[0]):
                    if t[j] > t[i] + w:
                        break
                    bij_max = ra_vub[k] * (t[j] - t[i]) - (data[j, k] - data[i, k])
                    bij_min = -ra_vlb[k] * (t[j] - t[i]) + (data[j, k] - data[i, k])
                    b.append(bij_max)
                    b.append(bij_min)
                    aij_max = np.zeros(2 * data.shape[0])
                    aij_min = np.zeros(2 * data.shape[0])
                    aij_max[j], aij_max[i] = 1, -1
                    aij_max[j + data.shape[0]], aij_max[i + data.shape[0]] = -1, 1
                    A.append(aij_max)
                    aij_min[j], aij_min[i] = -1, 1
                    aij_min[j + data.shape[0]], aij_min[i + data.shape[0]] = 1, -1
                    A.append(aij_min)
                    if i >= 1:
                        bij_max = ra_aub[k] * (t[j] - t[i]) - (data[j, k] - data[i, k]) / (t[j] - t[i]) + (
                                data[i, k] - data[i - 1, k]) / (t[i] - t[i - 1])
                        tmp1, tmp2 = 1 / (t[j] - t[i]), 1 / (t[i] - t[i - 1])
                        aij_max = np.zeros(2 * data.shape[0])
                        aij_max[j], aij_max[j + data.shape[0]] = tmp1, -tmp1
                        aij_max[i], aij_max[i + data.shape[0]] = -tmp1 - tmp2, tmp1 + tmp2
                        aij_max[i - 1], aij_max[i - 1 + data.shape[0]] = tmp2, -tmp2
                        b.append(bij_max)
                        A.append(aij_max)
                        aij_min = np.zeros(2 * data.shape[0])
                        bij_min = -ra_alb[k] * (t[j] - t[i]) + (data[j, k] - data[i, k]) / (t[j] - t[i]) - (
                                data[i, k] - data[i - 1, k]) / (t[i] - t[i - 1])
                        aij_min[j], aij_min[j + data.shape[0]] = -tmp1, tmp1
                        aij_min[i], aij_min[i + data.shape[0]] = tmp1 + tmp2, -tmp1 - tmp2
                        aij_min[i - 1], aij_min[i - 1 + data.shape[0]] = -tmp2, tmp2
                        b.append(bij_min)
                        A.append(aij_min)

            A = np.array(A)
            b = np.array(b)
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            # print(res.message)
            db.dataframe.loc[s:min(s + size, db.dataframe.shape[0]), ra[k]] = (res.x[:data.shape[0]] -
                                                                               res.x[data.shape[0]:]) + data[:, k]
            s += int((1 - overlapping_ratio) * size)
            # print(s)


# variance constraints based data cleaning
def variance_constraint_clean(db: DataBase, variance_constraints: dict, ra: list, t_attr, w=10, beta=0.5):
    # Assume that the timestamps are regular
    assert t_attr in db.dataframe.columns
    for attr in ra:
        assert attr in db.dataframe.columns
        assert attr in variance_constraints.keys()
        sequence = np.array(db.dataframe[attr])
        # Assume the first w points don't violate variance constraints
        assert np.var(sequence[: w]) <= variance_constraints[attr]
        for k in range(w-1, db.dataframe.shape[0] - 2*w):
            repair_sum = 0
            weight_sum = 0
            # candidate = []
            for i in range(k - w + 1, k + 1):
                num = i - (k - w + 1)
                weight_sum += pow(beta, num)
                if np.var(sequence[i: i+w]) <= variance_constraints[attr]:
                    # candidate.append(sequence[k])
                    repair_sum += pow(beta, num) * sequence[k]
                else:
                    # print(np.var(sequence[i: i+w]))
                    l1_sum = np.sum(sequence[i: i+w]) - sequence[k]
                    l2_sum = np.sum(sequence[i: i+w] * sequence[i: i+w])-sequence[k]*sequence[k]
                    x1, x2 = solve_quad(w-1, -2*l1_sum, w*l2_sum-l1_sum*l1_sum-w*w*variance_constraints[attr])
                    if x1 is None:
                        continue
                    elif abs(x1 - sequence[k]) > abs(x2 - sequence[k]):
                        # candidate.append(x2)
                        repair_sum += pow(beta, num) * x2

                    else:
                        # candidate.append(x1)
                        repair_sum += pow(beta, num) * x1

            repair = repair_sum / weight_sum
            if sequence[k] != repair:
                sequence[k] = repair
        db.dataframe[attr] = sequence


def speed_plus_acceleration_constraint_clean_local(db: DataBase, speed_constraints, acceleration_constraints,
                                             ra: list, t_attr, w=100):
    # Assume that the timestamps are regular
    assert t_attr in db.dataframe.columns
    for attr in ra:
        assert attr in db.dataframe.columns
        assert attr in speed_constraints.keys()
        assert attr in acceleration_constraints.keys()
    ra_range_v = [speed_constraints[attr] for attr in ra]
    ra_range_a = [acceleration_constraints[attr] for attr in ra]
    ra_vlb = np.array([x[0] for x in ra_range_v])
    ra_vub = np.array([x[1] for x in ra_range_v])
    ra_alb = np.array([x[0] for x in ra_range_a])
    ra_aub = np.array([x[1] for x in ra_range_a])
    data = np.array(db.dataframe[ra])
    t = np.array(db.dataframe[t_attr])
    for k in range(2, db.dataframe.shape[0] - 1):
        X_k_min, X_k_max = [], []
        x_k_min_v = ra_vlb * (t[k] - t[k - 1]) + data[k - 1]
        x_k_max_v = ra_vub * (t[k] - t[k - 1]) + data[k - 1]
        x_k_min_a = (ra_alb * (t[k] - t[k - 1]) + (data[k - 1] - data[k - 2])/(t[k - 1] - t[k - 2])) * \
                    (t[k] - t[k - 1]) + data[k - 1]
        x_k_max_a = (ra_aub * (t[k] - t[k - 1]) + (data[k - 1] - data[k - 2]) / (t[k - 1] - t[k - 2])) * \
                    (t[k] - t[k - 1]) + data[k - 1]
        x_k_min = np.where(x_k_min_v > x_k_min_a, x_k_min_v, x_k_min_a)
        x_k_max = np.where(x_k_max_v > x_k_max_a, x_k_max_v, x_k_max_a)
        for i in range(k + 1, db.dataframe.shape[0]):
            if t[i] > t[k] + w:
                break
            z_k_i_a_min = (data[k - 1] * (t[i] - t[k]) - (ra_aub * ((t[i] - t[k]) ** 2) - data[i]) * (
                    t[k] - t[k - 1])) / (t[i] - t[k - 1])
            z_k_i_a_max = (data[k - 1] * (t[i] - t[k]) - (ra_alb * ((t[i] - t[k]) ** 2) - data[i]) * (
                    t[k] - t[k - 1])) / (t[i] - t[k - 1])
            z_k_i_s_max = -ra_vlb * (t[i] - t[k]) + data[i]
            z_k_i_s_min = -ra_vub * (t[i] - t[k]) + data[i]
            X_k_min.append(np.where(z_k_i_a_min < z_k_i_s_min, z_k_i_a_min, z_k_i_s_min))
            X_k_max.append(np.where(z_k_i_a_max > z_k_i_s_max, z_k_i_a_max, z_k_i_s_max))
        x_k_mid = np.median(np.array([data[k]] + X_k_min + X_k_max), axis=0)
        x_k_mid = np.where(x_k_mid > x_k_max, x_k_max, x_k_mid)
        x_k_mid = np.where(x_k_mid < x_k_min, x_k_min, x_k_mid)
        data[k] = x_k_mid
    db.dataframe[ra] = data


class IMR:
    def __init__(self, db_dirty: DataBase, db_label: DataBase, ra: list, p, delta: dict, max_iter=100):
        self.db_dirty = db_dirty
        self.db_label = db_label
        self.ra = ra
        assert db_dirty.dataframe.shape[0] == db_label.dataframe.shape[0]
        for attr in ra:
            assert attr in db_dirty.dataframe.columns
            assert attr in db_label.dataframe.columns
        self.p = p
        self.delta = delta
        self.max_iter = max_iter

    def repair(self):
        for attr in self.ra:
            x = np.array(self.db_dirty.dataframe[attr])
            y = np.array(self.db_label.dataframe[attr])
            residual = y - x
            residual_last = np.array(residual, copy=True)
            k = 0
            while k < self.max_iter:
                k += 1
                ar_model = AutoReg(residual_last, self.p).fit()
                residual_p = np.append(residual_last[:self.p], ar_model.predict(start=self.p))
                residual_p = np.where((residual == 0) & (np.abs(residual_p - residual_last) > self.delta[attr]),
                                      residual_p, 0)
                if (residual_p == 0).all():
                    break
                index = np.argmin(np.abs(residual_p))
                residual_last[index] = residual_p[index]
            result = residual_last + x
            self.db_dirty.dataframe[attr] = result


def ewma(db: DataBase, ra: list, alpha=None, sliding=None):
    assert alpha is not None or sliding is not None
    for attr in ra:
        assert attr in db.dataframe.columns
    if alpha is not None:
        alpha = alpha
    else:
        alpha = 2 / (sliding + 1)
    data = np.array(db.dataframe[ra])
    result = [data[0]]
    last = data[0]
    for i in range(1, data.shape[0]):
        last = data[i] * alpha + (1 - alpha) * last
        result.append(last)
    db.dataframe[ra] = np.array(result)


class Clean4MTSNR(Clean4MTS):
    def __init__(self, db: DataBase, regression_rule: list, speed_constraint: dict, temporal_attr: str, src: list,
                 target: list, delta_w=12, beta=0.5, max_window_size=500, print_repair_message=False,
                 method='studGA', NIND=50, MAXGEN=500, trappedValue=1e-6, w1=None, w2=None):
        super().__init__(db=db, regression_rule=regression_rule, speed_constraint=speed_constraint,
                         temporal_attr=temporal_attr, src=src, target=target, delta_w=delta_w, beta=beta,
                         max_window_size=max_window_size, print_repair_message=print_repair_message,
                         method='studGA', NIND=NIND, MAXGEN=MAXGEN, trappedValue=trappedValue, w1=w1, w2=w2)

    def violation_detection(self):
        xtest, ytest = np.array(self.dataframe[self.src]), \
            np.array(self.dataframe[self.target]).reshape([self.shape[0], ])
        if self.rr[0][0] == 'bayesian':
            bias = np.array(ytest - self.rr[0][1].predict(np.vander(xtest.T[0], N=4)))
        else:
            bias = np.array(ytest - self.rr[0][1].predict(xtest))
        bias += self.rr[2]
        vd = np.abs(bias) - self.rr[1]
        vi = np.where(vd > 0)[0]
        s = None
        last = None
        end = vi[-1]
        for i in vi:
            if s is None:
                s = i
                last = i
                continue
            elif i == last + 1:
                last += 1
            else:
                self.vtw_index.append((s, last))
                s = i
                last = i

            if i == end:
                self.vtw_index.append((s, i))

    def data_cleaning(self):
        self.violation_detection()
        super().data_repair()


