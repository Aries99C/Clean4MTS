# -*- coding:utf-8 -*-
# @Author : Genglong Li
# @Time : 2023/4/3 9:58
from DQDiscovery.LearnCRR.core import DataBase
import geatpy as ea
import numpy as np
from collections import deque


class Clean4MTS:
    def __init__(self, db: DataBase, regression_rule: list, speed_constraint: dict, temporal_attr: str, src: list,
                 target: list, delta_w=12, beta=0.5, max_window_size=500, print_repair_message=False,
                 method='studGA', NIND=50, MAXGEN=500, trappedValue=1e-6, w1=None, w2=None):
        if temporal_attr in src:
            self.dataframe = db.dataframe[src + target].reset_index()
        else:
            attrs = src + target
            attrs.append(temporal_attr)
            self.dataframe = db.dataframe[attrs].reset_index()
        # Assume that the timestamp attribute is not in X or {Y}
        self.var_omega = src + target
        assert temporal_attr not in self.var_omega
        for attr in speed_constraint.keys():
            assert attr in self.var_omega
        self.shape = self.dataframe.shape
        self.sc = speed_constraint
        self.rr = regression_rule
        self.src = src
        self.target = target
        self.temporal_attr = temporal_attr
        self.delta_w = delta_w
        self.beta = beta
        self.vtw_index = []
        self.max_window_size = max_window_size
        # Parameters for Data Repairing
        self.method = method
        self.w1 = w1
        self.w2 = w2
        self.NIND = NIND
        self.MAXGEN = MAXGEN
        self.trappedValue = trappedValue
        self.print_repair_message = print_repair_message
        # Speed Constraints Preprocessing
        var_omega_vrange = [speed_constraint[attr] for attr in self.var_omega]
        self.var_omega_vlb = np.array([x[0] for x in var_omega_vrange])
        self.var_omega_vub = np.array([x[1] for x in var_omega_vrange])

    def quantify_deviation_through_sliding_window(self, index, max_window_size):
        s_index = index
        e_index = min(index + max_window_size, self.shape[0])
        xtest, ytest = np.array(self.dataframe.loc[s_index:e_index-1, self.src]), \
            np.array(self.dataframe.loc[s_index:e_index-1, self.target]).reshape([e_index-s_index, ])
        if self.rr[0][0] == 'bayesian':
            bias = np.array(ytest - self.rr[0][1].predict(np.vander(xtest.T[0], N=4)))
        else:
            bias = np.array(ytest - self.rr[0][1].predict(xtest))
        bias += self.rr[2]
        return np.abs(bias) - self.rr[1]

    def violation_localization(self, profile):
        l = len(profile)
        assert l % 2 == 1
        gamma = int((l + 1) / 2)
        # print(profile)
        start_array = -np.ones(gamma, dtype=int)
        q = deque()
        q.append(0)
        while q:
            start_index = q.popleft()
            gain = 0
            for i in range(start_index, l):
                gain += profile[i][2]
                if i % 2 == 1:
                    continue
                else:
                    if gain > 0 and start_array[int(i/2)] == -1:
                        start_array[int(i/2)] = int(start_index/2)
                        if i == l - 1:
                            e_index = int(i / 2)
                            while 1:
                                s_index = start_array[e_index]
                                self.vtw_index.append((profile[s_index*2][0], profile[e_index*2][1]))

                                e_index = s_index - 1
                                if e_index < 0:
                                    return
                        else:
                            q.append(i + 2)

    def violation_profiling(self, max_window_size):
        index = 0
        profile = []
        while index < self.shape[0]:
            cur_dev = self.quantify_deviation_through_sliding_window(index, max_window_size)
            vtis = np.where(cur_dev > 0)[0]
            if vtis.shape[0] == 0:
                if len(profile) != 0:
                    self.violation_localization(profile)
                index += max_window_size
                continue
            ts_positive = None
            last_ti = None
            increment = max_window_size
            for ti in vtis:
                if ti + self.delta_w > max_window_size - 1:
                    if last_ti < ti - self.delta_w:
                        triple_positive = [ts_positive + index, last_ti + index, np.sum(
                            1 - np.exp(-cur_dev[ts_positive: last_ti + 1] / self.rr[1] * self.beta))]
                        profile.append(triple_positive)
                        self.violation_localization(profile)
                        profile = []
                        increment = ti
                        break

                    else:
                        if len(profile) == 0:
                            increment = ti
                        else:
                            increment = profile[0][0]
                            profile = []
                            break

                elif ts_positive is None:
                    last_ti = ti
                    ts_positive = ti
                    continue

                elif ti == vtis[-1]:
                    if last_ti < ti - self.delta_w:
                        triple_positive = [ts_positive + index, last_ti + index, np.sum(
                            1 - np.exp(-cur_dev[ts_positive:  last_ti + 1] / self.rr[1] * self.beta))]
                        profile.append(triple_positive)
                        self.violation_localization(profile)
                        profile = []
                    elif last_ti == ti - 1:
                        triple_positive = [ts_positive + index, ti + index, np.sum(
                            1 - np.exp(-cur_dev[ts_positive:  ti + 1] / self.rr[1] * self.beta))]
                        profile.append(triple_positive)
                        self.violation_localization(profile)
                        profile = []

                    else:
                        triple_positive = [ts_positive + index, last_ti + index, np.sum(
                            1 - np.exp(-cur_dev[ts_positive: last_ti + 1] / self.rr[1] * self.beta))]

                        triple_negative = [last_ti + 1 + index, ti - 1 + index, np.sum(
                            cur_dev[last_ti + 1: ti] / self.rr[1])]
                        profile.append(triple_positive)
                        profile.append(triple_negative)
                        triple_positive = [ti + index, ti + index, np.sum(
                            1 - np.exp(-cur_dev[ti: ti + 1] / self.rr[1] * self.beta))]
                        profile.append(triple_positive)
                        self.violation_localization(profile)
                        profile = []
                    break
                else:
                    if last_ti == ti - 1:
                        last_ti = ti
                        continue
                    else:
                        if last_ti > ti - self.delta_w:
                            triple_positive = [ts_positive + index, last_ti + index, np.sum(
                                1 - np.exp(-cur_dev[ts_positive: last_ti + 1] / self.rr[1] * self.beta))]

                            triple_negative = [last_ti + 1 + index, ti - 1 + index, np.sum(
                                cur_dev[last_ti + 1: ti] / self.rr[1])]
                            profile.append(triple_positive)
                            profile.append(triple_negative)
                            last_ti = ti
                            ts_positive = ti

                        else:
                            triple_positive = [ts_positive + index, last_ti + index, np.sum(
                                1 - np.exp(-cur_dev[ts_positive: last_ti + 1] / self.rr[1] * self.beta))]
                            profile.append(triple_positive)
                            self.violation_localization(profile)
                            profile = []
                            ts_positive = ti
                            last_ti = ti
            index += increment

    def prophet_generator(self, data_instance, lb, ub, weight, n=None):
        dim = len(self.var_omega)
        prophet = []
        if n is None:
            n = 3 * dim
        w = np.random.normal(loc=0, scale=1 / 3, size=n)
        sampleList = [x for x in range(dim)]
        weight = weight / np.sum(weight)
        var_omega_i = np.random.choice(sampleList, n, p=weight)

        for i in range(n):
            p = np.array(data_instance, copy=True)
            j = var_omega_i[i]
            p[j] = (ub[j] + lb[j])/2 + (ub[j] - lb[j])/2 * max(w[i], 1)
            prophet.append(p)
        return np.array(prophet)

    def repair_violation_single(self, s, e):
        assert s > 0
        assert e < self.shape[0] - 1
        assert s == e - 1
        p = np.array(self.dataframe.loc[s - 1: e, self.var_omega])
        t = np.array(self.dataframe.loc[s - 1: e, self.temporal_attr].diff())[1:, ]
        if t[0] < t[1]: # Calculate the upper and lower boundaries for all attributes related to rr
            # prophet = p[0]
            lb_pri = p[0] + t[0] * self.var_omega_vlb
            ub_pri = p[0] + t[0] * self.var_omega_vub
            lb_sub = p[-1] - t[-1] * self.var_omega_vub
            ub_sub = p[-1] - t[-1] * self.var_omega_vlb
            lb = np.where((lb_pri < lb_sub) & (lb_sub <= p[0]), lb_sub, lb_pri)
            ub = np.where((ub_pri > ub_sub) & (ub_sub >= p[0]), ub_sub, ub_pri)
            p_pri = p[0: 1]

        else:
            lb_sub = p[0] + t[0] * self.var_omega_vlb
            ub_sub = p[0] + t[0] * self.var_omega_vub
            lb_pri = p[-1] - t[-1] * self.var_omega_vub
            ub_pri = p[-1] - t[-1] * self.var_omega_vlb
            lb = np.where((lb_pri < lb_sub) & (lb_sub <= p[-1]), lb_sub, lb_pri)
            ub = np.where((ub_pri > ub_sub) & (ub_sub >= p[-1]), ub_sub, ub_pri)
            p_pri = p[-1:]

        difference = np.abs(p[1] - (ub + lb) / 2) / (ub - lb)
        if (difference <= 1 / 2).all():
            if self.print_repair_message:
                print('index:', s)
                print('Speed Diff', difference)
                print('Value no repair', p[1])
            return
        prophet = np.concatenate(
            (self.prophet_generator(data_instance=p[1], lb=lb, ub=ub, weight=difference, n=self.NIND - 1), p_pri),
            axis=0)
        op = OptimizeProblemPlus(data_instance=p[1], lb=lb, ub=ub, rr=self.rr, beta=self.beta, w1=self.w1, w2=self.w2)
        if self.print_repair_message:
            print('index:', s)
            print('Speed Diff', difference)
            print('Value before repair', p[1])

        p_repaired = solve_optimization(problem=op, prophet=prophet, method=self.method, NIND=self.NIND,
                                        MAXGEN=self.MAXGEN, trappedValue=self.trappedValue)
        if self.print_repair_message:
            print('Value after repair', p_repaired)
        self.dataframe.loc[s, self.var_omega] = p_repaired

    def repair_violation_continuous(self, s, e):
        assert s > 0
        assert e < self.shape[0] - 1
        assert s < e - 1
        width = e - s
        pbf = np.array(self.dataframe.loc[s - 1: s + int(width / 2) - 1, self.var_omega])
        tbf = np.array(self.dataframe.loc[s - 1: s + int(width / 2) - 1, self.temporal_attr])
        paf = np.array(self.dataframe.loc[e - int(width / 2): e, self.var_omega])
        taf = np.array(self.dataframe.loc[e - int(width / 2): e, self.temporal_attr])
        for i in range(1, int(width / 2) + 1):
            # prophet_bf = pbf[i-1]
            lb_bf_pri = pbf[i - 1] + (tbf[i] - tbf[i - 1]) * self.var_omega_vlb
            ub_bf_pri = pbf[i - 1] + (tbf[i] - tbf[i - 1]) * self.var_omega_vub
            lb_bf_sub = paf[-i] + (tbf[i] - taf[-i]) * self.var_omega_vub
            ub_bf_sub = paf[-i] + (tbf[i] - taf[-i]) * self.var_omega_vlb
            lb_bf = np.where((lb_bf_pri < lb_bf_sub) & (lb_bf_sub <= pbf[i - 1]), lb_bf_sub, lb_bf_pri)
            ub_bf = np.where((ub_bf_pri > ub_bf_sub) & (ub_bf_sub >= pbf[i - 1]), ub_bf_sub, ub_bf_pri)
            difference = np.abs(pbf[i] - (ub_bf + lb_bf) / 2) / ((ub_bf - lb_bf)/2)
            prophet_bf = np.concatenate(
                (self.prophet_generator(data_instance=pbf[i], lb=lb_bf, ub=ub_bf,
                                        weight=difference, n=self.NIND - 1), pbf[i-1: i]), axis=0)
            op_bf = OptimizeProblemPlus(data_instance=pbf[i], lb=lb_bf, ub=ub_bf, rr=self.rr, beta=self.beta,
                                        w1=self.w1,
                                        w2=self.w2)

            if self.print_repair_message:
                print('index:', s + i - 1)
                print('Value before repair', pbf[i])
            if (difference <= 1 / 2).all():
                if self.print_repair_message:
                    print('Speed Diff', difference)
                    print('Value no repair', pbf[i])
            else:
                p_repaired = solve_optimization(problem=op_bf, prophet=prophet_bf, method=self.method, NIND=self.NIND,
                                                MAXGEN=self.MAXGEN, trappedValue=self.trappedValue)
                if self.print_repair_message:
                    print('Value after repair', p_repaired)
                pbf[i] = p_repaired


            # prophet_af = paf[-i]
            lb_af_pri = paf[-i] + (taf[-i - 1] - taf[-i]) * self.var_omega_vub
            ub_af_pri = paf[-i] + (taf[-i - 1] - taf[-i]) * self.var_omega_vlb
            lb_af_sub = pbf[i - 1] + (taf[-i - 1] - tbf[i - 1]) * self.var_omega_vlb
            ub_af_sub = pbf[i - 1] + (taf[-i - 1] - tbf[i - 1]) * self.var_omega_vub
            lb_af = np.where((lb_af_pri < lb_af_sub) & (lb_af_sub <= paf[-i]), lb_af_sub, lb_af_pri)
            ub_af = np.where((ub_af_pri > ub_af_sub) & (ub_af_sub >= paf[-i]), ub_af_sub, ub_af_pri)
            difference = np.abs(paf[-i - 1] - (ub_af + lb_af) / 2) / ((ub_af - lb_af)/2)
            prophet_af = np.concatenate(
                (self.prophet_generator(data_instance=paf[-i - 1], lb=lb_af, ub=ub_af,
                                        weight=difference, n=self.NIND - 1), paf[-i: -i+1]), axis=0)
            op_af = OptimizeProblemPlus(data_instance=paf[-i - 1], lb=lb_af, ub=ub_af, rr=self.rr, beta=self.beta,
                                        w1=self.w1, w2=self.w2)
            if self.print_repair_message:
                print('index:', e - i)
                print('Value before repair', paf[-i - 1])

            if (difference <= 1 / 2).all():
                if self.print_repair_message:
                    print('Speed Diff', difference)
                    print('Value no repair', paf[-i - 1])
            else:
                p_repaired = solve_optimization(problem=op_af, prophet=prophet_af, method=self.method, NIND=self.NIND,
                                                MAXGEN=self.MAXGEN, trappedValue=self.trappedValue)
                if self.print_repair_message:
                    print('Value after repair', p_repaired)
                paf[-i - 1] = p_repaired

        self.dataframe.loc[s - 1: s + int(width / 2) - 1, self.var_omega] = pbf
        self.dataframe.loc[e - int(width / 2): e, self.var_omega] = paf
        if width % 2 != 0:
            self.repair_violation_single(s + int(width / 2), s + int(width / 2) + 1)

    def data_repair(self):
        for tw in self.vtw_index:
            s = tw[0]
            e = tw[1] + 1
            if e - s == 1:
                self.repair_violation_single(s, e)
            elif e - s > 1:
                self.repair_violation_continuous(s, e)

    def data_cleaning(self):
        self.violation_profiling(max_window_size=self.max_window_size)
        self.data_repair()


class OptimizeProblemPlus(ea.Problem):
    def __init__(self, data_instance: np.ndarray, lb: np.ndarray, ub: np.ndarray, rr, beta=1, w1=None, w2=None,
                 repair_attr=None):
        self.Dim = lb.shape[0]  # 初始化Dim（决策变量维数）
        self.datainstance = data_instance.reshape([self.Dim, ])  # The last attribute is the target
        self.rr = rr
        if w1 is None:
            w1 = self.Dim * 3
        if w2 is None:
            w2 = self.Dim
        if repair_attr is None:
            self.repair_attr = np.arange(self.Dim)
            self.fixed_attr = np.array([])
            self.lb = lb - data_instance  # lower boundary for each decision variable
            self.ub = ub - data_instance  # upper boundary for each decision variable
        else:
            self.repair_attr = np.array(repair_attr) # The attributes where data repair may take place
            self.fixed_attr = np.setdiff1d(np.arange(self.Dim), self.repair_attr)
            self.Dim = repair_attr.shape[0]
            self.lb = lb[repair_attr] - data_instance[repair_attr]  # lower boundary for each decision variable
            self.ub = ub[repair_attr] - data_instance[repair_attr]  # upper boundary for each decision variable
        self.w1 = w1
        self.w2 = w2
        M = 1  # Initialize the dimension
        name = 'ea_problem'
        maxormins = [1]  # One object to be minimized (1 means minimization while -1 means maximization
        varTypes = [0] * self.Dim

        self.beta = beta
        lbin = [1] * self.Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * self.Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        w = []
        for i in range(self.Dim):
            max_ = max(ub[i], data_instance[i])
            min_ = min(lb[i], data_instance[i])
            if max_ > min_:
                w.append(1 / (max_ - min_))
            elif max_ == min_:
                w.append(0)
            else:
                raise ValueError('The bounds seem erroneous！')
        self.w = np.array([w]).T  # 列向量
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            self.Dim,
                            varTypes,
                            self.lb,
                            self.ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars):  # 目标函数
        data = np.copy(Vars)
        for fi in self.fixed_attr:
            inserted = np.zeros(Vars.shape[1])
            data = np.insert(data, fi, inserted, axis=1)

        src = data[:, 0:-1] + self.datainstance[0: -1]
        target = data[:, -1] + self.datainstance[-1]
        l = Vars.shape[0]
        if self.rr[0][0] == 'bayesian':
            bias = np.array(target - self.rr[0][1].predict(np.vander(src.T[0], N=4)))
        else:
            bias = np.array(target - self.rr[0][1].predict(src))
        bias += self.rr[2]
        deg = np.abs(bias) - self.rr[1]
        deg[deg <= 0] = deg[deg <= 0] / self.rr[1]
        bounds = (self.ub - self.lb) / 1000
        count = np.sum(np.where(Vars > bounds, 1, 0), axis=1).reshape(Vars.shape[0], 1) / self.Dim
        deg[deg > 0] = (1 - np.exp(-deg[deg > 0] / self.rr[1] * self.beta)) + self.w1 + self.w2
        f = np.matmul(np.abs(Vars), self.w) / self.Dim * self.w1 + self.w2 * count + deg.reshape([l, 1])
        return f


def solve_optimization(problem: OptimizeProblemPlus, prophet, method, NIND=50,
                       MAXGEN=500, trappedValue=1e-6, maxTrappedCount=15):
    # Construct the evolutionary algorithm
    if method == 'DE_rand_1_bin':
        algorithm = ea.soea_DE_rand_1_bin_templet(
            problem,
            ea.Population(Encoding='RI', NIND=NIND),  # 实数编码。 种群个体50
            MAXGEN=MAXGEN,  # 最大进化代数。
            logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
            trappedValue=trappedValue,  # 单目标优化陷入停滞的判断阈值。
            maxTrappedCount=maxTrappedCount)  # 进化停滞计数器最大上限值。

    elif method == 'ES_miu_plus_lambda':
        algorithm = ea.soea_ES_miu_plus_lambda_templet(
            problem,
            ea.Population(Encoding='RI', NIND=NIND),  # 实数编码。 种群个体50
            MAXGEN=MAXGEN,  # 最大进化代数。
            logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
            trappedValue=trappedValue,  # 单目标优化陷入停滞的判断阈值。
            maxTrappedCount=maxTrappedCount)  # 进化停滞计数器最大上限值。

    else:
        algorithm = ea.soea_studGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=NIND),  # 实数编码。 种群个体50
            MAXGEN=MAXGEN,  # 最大进化代数。
            logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
            trappedValue=trappedValue,  # 单目标优化陷入停滞的判断阈值。
            maxTrappedCount=maxTrappedCount)  # 进化停滞计数器最大上限值。

    # Prior Knowledge
    prophetVars = prophet - problem.datainstance  # 假设已知[0.4, 0.2, 0.4]为一组比较优秀的变量。
    # Solve the optimization problem
    res = ea.optimize(algorithm,
                      prophet=prophetVars,
                      verbose=True,
                      drawing=0,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    result = problem.datainstance
    for attr_i in range(problem.repair_attr.shape[0]):
        result[problem.repair_attr[attr_i]] = res['Vars'][0][attr_i] + result[problem.repair_attr[attr_i]]
    return result


class Clean4MTSPlus(Clean4MTS):
    def __init__(self, db: DataBase, regression_rule: list, speed_constraint: dict, acceleration_constraint: dict,
                 temporal_attr: str, src: list, target: list, delta_w=6, beta=1, max_window_size=1000, method='studGA',
                 NIND=50, MAXGEN=500, trappedValue=1e-6, w1=None, w2=None, print_repair_message=False):
        super().__init__(db=db, regression_rule=regression_rule, speed_constraint=speed_constraint,
                         temporal_attr=temporal_attr, src=src, target=target, delta_w=delta_w, beta=beta,
                         max_window_size=max_window_size, print_repair_message=print_repair_message,
                            method='studGA', NIND=NIND, MAXGEN=MAXGEN, trappedValue=trappedValue, w1=w1, w2=w2)

        for attr in acceleration_constraint.keys():
            assert attr in self.var_omega
        var_omega_arange = [acceleration_constraint[attr] for attr in self.var_omega]
        self.var_omega_alb = np.array([x[0] for x in var_omega_arange])
        self.var_omega_aub = np.array([x[1] for x in var_omega_arange])
        assert (self.var_omega_aub > self.var_omega_alb).all()

    def repair_violation_single(self, s, e):
        assert s > 1
        assert e < self.shape[0] - 1
        assert s == e - 1
        p = np.array(self.dataframe.loc[s - 2: e + 1, self.var_omega])
        t = np.array(self.dataframe.loc[s - 2: e + 1, self.temporal_attr].diff())[1:, ]

        if t[1] <= t[2]:
            vlb_pri = (p[1] - p[0]) / t[0] + self.var_omega_alb * t[1]
            vlb_pri = np.where(self.var_omega_vlb > vlb_pri, self.var_omega_vlb, vlb_pri)
            vlb_pri = np.where(vlb_pri > 0, 0, vlb_pri)
            vub_pri = (p[1] - p[0]) / t[0] + self.var_omega_aub  * t[1]
            vub_pri = np.where(self.var_omega_vub < vub_pri, self.var_omega_vub, vub_pri)
            vub_pri = np.where(vub_pri < 0, 0, vub_pri)
            lb_pri = p[1] + t[1] * vlb_pri
            ub_pri = p[1] + t[1] * vub_pri
            vlb_sub = self.var_omega_alb * t[3] - (p[4] - p[3]) / t[3]
            vub_sub = self.var_omega_aub * t[3] - (p[4] - p[3]) / t[3]
            vlb_sub = np.where(-self.var_omega_vub > vlb_sub, -self.var_omega_vub, vlb_sub)
            vub_sub = np.where(-self.var_omega_vlb < vlb_sub, -self.var_omega_vlb, vub_sub)

            lb_sub = p[3] + t[2] * vlb_sub
            ub_sub = p[3] + t[2] * vub_sub
            lb = np.where((lb_pri < lb_sub) & (lb_sub <= p[1]), lb_sub, lb_pri)
            ub = np.where((ub_pri > ub_sub) & (ub_sub >= p[1]), ub_sub, ub_pri)
            p_pri = p[0: 2]

        else:
            vlb_pri = self.var_omega_alb * t[3] - (p[4] - p[3]) / t[3]
            vub_pri = self.var_omega_aub * t[3] - (p[4] - p[3]) / t[3]
            vlb_pri = np.where(-self.var_omega_vub > vlb_pri, -self.var_omega_vub, vlb_pri)
            vub_pri = np.where(-self.var_omega_vlb < vlb_pri, -self.var_omega_vlb, vub_pri)
            vlb_pri = np.where(vlb_pri > 0, 0, vlb_pri)
            vub_pri = np.where(vub_pri < 0, 0, vub_pri)
            lb_pri = p[3] + t[2] * vlb_pri
            ub_pri = p[3] + t[2] * vub_pri

            vlb_sub = (p[1] - p[0]) / t[0] + self.var_omega_alb * t[1]
            vlb_sub = np.where(self.var_omega_vlb > vlb_sub, self.var_omega_vlb, vlb_sub)
            vub_sub = (p[1] - p[0]) / t[0] + self.var_omega_aub * t[1]
            vub_sub = np.where(self.var_omega_vub < vub_sub, self.var_omega_vub, vub_sub)
            lb_sub = p[1] + t[1] * vlb_sub
            ub_sub = p[1] + t[1] * vub_sub
            p_pri = p[3:]
            lb = np.where((lb_pri < lb_sub) & (lb_sub <= p[-1]), lb_sub, lb_pri)
            ub = np.where((ub_pri > ub_sub) & (ub_sub >= p[-1]), ub_sub, ub_pri)

        difference = np.abs(p[2] - (ub + lb) / 2) / (ub - lb)
        if (difference <= 1 / 2).all():
            if self.print_repair_message:
                print('index:', s)
                print('Speed Diff', difference)
                print('Value no repair', p[2])
                return
        prophet = np.concatenate(
                (super().prophet_generator(data_instance=p[2], lb=lb, ub=ub, weight=difference, n=self.NIND - 2), p_pri),
                axis=0)

        op = OptimizeProblemPlus(data_instance=p[2], lb=lb, ub=ub, rr=self.rr, beta=self.beta, w1=self.w1, w2=self.w2)
        if self.print_repair_message:
            print('index:', s)
            print('Speed Diff', difference)
            print('Value before repair', p[1])

        p_repaired = solve_optimization(problem=op, prophet=prophet, method=self.method, NIND=self.NIND,
                                        MAXGEN=self.MAXGEN, trappedValue=self.trappedValue)
        if self.print_repair_message:
            print('Value after repair', p_repaired)
        self.dataframe.loc[s, self.var_omega] = p_repaired

    def repair_violation_continuous(self, s, e):
        assert s > 0
        assert e < self.shape[0] - 1
        assert s < e - 1
        width = e - s
        pbf = np.array(self.dataframe.loc[s - 2: s + int(width / 2) - 1, self.var_omega])
        tbf = np.array(self.dataframe.loc[s - 2: s + int(width / 2) - 1, self.temporal_attr])
        paf = np.array(self.dataframe.loc[e - int(width / 2): e + 1, self.var_omega])
        taf = np.array(self.dataframe.loc[e - int(width / 2): e + 1, self.temporal_attr])
        for i in range(2, pbf.shape[0]):

            vlb_pri = (pbf[i - 1] - pbf[i - 2]) / (tbf[i - 1] - tbf[i - 2]) + self.var_omega_alb * (tbf[i] - tbf[i - 1])
            vlb_pri = np.where(self.var_omega_vlb > vlb_pri, self.var_omega_vlb, vlb_pri)
            vlb_pri = np.where(vlb_pri > 0, 0, vlb_pri)
            lb_bf = pbf[i - 1] + (tbf[i] - tbf[i - 1]) * vlb_pri

            vub_pri = (pbf[i - 1] - pbf[i - 2]) / (tbf[i - 1] - tbf[i - 2]) + self.var_omega_aub * (tbf[i] - tbf[i - 1])
            vub_pri = np.where(self.var_omega_vub < vub_pri, self.var_omega_vub, vub_pri)
            vub_pri = np.where(vub_pri < 0, 0, vub_pri)
            ub_bf = pbf[i - 1] + (tbf[i] - tbf[i - 1]) * vub_pri

            difference = np.abs(pbf[i] - (ub_bf + lb_bf) / 2) / ((ub_bf - lb_bf) / 2)
            prophet_bf = np.concatenate(
                (super().prophet_generator(data_instance=pbf[i - 1], lb=lb_bf, ub=ub_bf,
                                        weight=difference, n=self.NIND - 2), pbf[i - 3: i - 1]), axis=0)
            op_bf = OptimizeProblemPlus(data_instance=pbf[i], lb=lb_bf, ub=ub_bf, rr=self.rr, beta=self.beta,
                                        w1=self.w1,
                                        w2=self.w2)
            if self.print_repair_message:
                print('index:', s + i - 2)
                print('Value before repair', pbf[i])
            if (difference <= 1 / 2).all():
                if self.print_repair_message:
                    print('Diff', difference)
                    print('Value no repair', pbf[i])
            else:
                p_repaired = solve_optimization(problem=op_bf, prophet=prophet_bf, method=self.method, NIND=self.NIND,
                                                MAXGEN=self.MAXGEN, trappedValue=self.trappedValue)
                if self.print_repair_message:
                    print('Value after repair', p_repaired)
                pbf[i] = p_repaired


            vub_pri = (paf[-i + 1] - paf[-i]) / (taf[-i + 1] - taf[-i]) - self.var_omega_alb * (taf[-i + 1] - taf[-i])
            vlb_pri = (paf[-i + 1] - paf[-i]) / (taf[-i + 1] - taf[-i]) - self.var_omega_aub * (taf[-i + 1] - taf[-i])
            vub_pri = np.where(self.var_omega_vub < vub_pri, self.var_omega_vub, vub_pri)
            vlb_pri = np.where(self.var_omega_vlb > vlb_pri, self.var_omega_vlb, vlb_pri)
            lb_af_pri = paf[-i] + (taf[-i - 1] - taf[-i]) * vub_pri
            ub_af_pri = paf[-i] + (taf[-i - 1] - taf[-i]) * vlb_pri
            lb_af = np.where(lb_af_pri > paf[-i], paf[-i], lb_af_pri)
            ub_af = np.where(ub_af_pri < paf[-i], paf[-i], ub_af_pri)

            difference = np.abs(paf[-i - 1] - (ub_af + lb_af) / 2) / ((ub_af - lb_af) / 2)

            prophet_af = np.concatenate(
                (super().prophet_generator(data_instance=paf[-i - 1], lb=lb_af, ub=ub_af,
                                        weight=difference, n=self.NIND - 2), paf[-i: i+2]), axis=0)
            op_af = OptimizeProblemPlus(data_instance=paf[-i - 1], lb=lb_af, ub=ub_af, rr=self.rr, beta=self.beta,
                                        w1=self.w1, w2=self.w2)
            if self.print_repair_message:
                print('index:', e - i + 1)
                print('Value before repair', paf[-i - 1])

            if (difference <= 1 / 2).all():
                if self.print_repair_message:
                    print('Speed Diff', difference)
                    print('Value no repair', paf[-i - 1])
            else:
                p_repaired = solve_optimization(problem=op_af, prophet=prophet_af, method=self.method, NIND=self.NIND,
                                                MAXGEN=self.MAXGEN, trappedValue=self.trappedValue)
                if self.print_repair_message:
                    print('Value after repair', p_repaired)
                paf[-i - 1] = p_repaired

        self.dataframe.loc[s - 1: s + int(width / 2) - 1, self.var_omega] = pbf[1:, :]
        self.dataframe.loc[e - int(width / 2): e, self.var_omega] = paf[:-1, :]
        if width % 2 != 0:
            self.repair_violation_single(s + int(width / 2), s + int(width / 2) + 1)

    def data_repair(self):
        for tw in self.vtw_index:
            s = tw[0]
            e = tw[1] + 1
            if e - s == 1:
                self.repair_violation_single(s, e)
            elif e - s > 1:
                self.repair_violation_continuous(s, e)

    def data_cleaning(self):
        super().violation_profiling(max_window_size=self.max_window_size)
        self.data_repair()

