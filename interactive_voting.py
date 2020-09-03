import os
import math
import numpy as np
import pandas as pd

from utils import data_loader
import utils.display as display

from sklearn.metrics.pairwise import haversine_distances
from scipy.stats import norm
from dask.distributed import Client
from dask import delayed


def find_sequence(c_s, weight, phi, i, k, a_i, trajectory_len):
    t_phi = [item.copy() for item in phi]
    # c_s['f'] = np.NINF
    # c_s['pre'] = None
    f = [np.NINF] * len(c_s.index)
    pre = [None] * len(c_s.index)
    # set f of first trajectory point
    s = c_s['t_p_i'].searchsorted(0, side='left')
    e = c_s['t_p_i'].searchsorted(0, side='right')
    # c_s.loc[0, 'f'] = weight[0] * c_s.loc[0, 'epsilon'].values
    f[s:e] = weight[0] * c_s.iloc[s:e, c_s.columns.get_loc('epsilon')]

    for s in range(a_i):
        if s != k:
            if 0 == i:
                f[s] = np.NINF
                # c_s.loc[(i, s), 'f'] = np.NINF
            else:
                t_phi[i - 1][:, s] = np.NINF

    j_pre_s = 0
    j_pre_e = c_s['t_p_i'].searchsorted(0, side='right')
    for j in range(1, trajectory_len):
        j_s = c_s['t_p_i'].searchsorted(j, side='left')
        j_e = c_s['t_p_i'].searchsorted(j, side='right')
        for s in range(j_e - j_s):
            # print(c_s.loc[j - 1, 'f'], t_phi[j - 1][:, s])
            last_f = f[j_pre_s: j_pre_e]
            f[j_s + s] = max(last_f + t_phi[j - 1][:, s])
            pre[j_s + s] = np.argmax(last_f + t_phi[j - 1][:, s])
            # c_s.loc[(j, s), 'f'] = max(c_s.loc[j - 1, 'f'] + t_phi[j - 1][:, s])
            # c_s.loc[(j, s), 'pre'] = np.argmax(c_s.loc[j - 1, 'f'] + t_phi[j - 1][:, s])
        j_pre_s = j_s
        j_pre_e = j_e

    r_list = []
    last_i = trajectory_len - 1
    lst_i_s = c_s['t_p_i'].searchsorted(last_i, side='left')
    c = np.argmax(f[lst_i_s:])
    for j in reversed(range(1, trajectory_len)):
        s = c_s['t_p_i'].searchsorted(j, side='left')
        e = c_s['t_p_i'].searchsorted(j, side='right')
        r_list.append(c)
        c = pre[s+c]
    r_list.append(c)

    return max(f[lst_i_s:]), list(reversed(r_list))


def traverse_trajectory_point(i, M, cur_vehicle, candidate_set, trajectory_len):
    # compute phi for current trajectory point
    phi = [item.copy() for item in M]
    # prepare trajectory location

    loc_i = cur_vehicle.iloc[[i], [cur_vehicle.columns.get_loc('latitude'),
                                   cur_vehicle.columns.get_loc('longitude')]].values
    loc_all = cur_vehicle.loc[cur_vehicle.index != i, ['latitude', 'longitude']].values
    # compute trajectory point distance

    tp_dis = 6371009 * haversine_distances(np.radians(loc_i), np.radians(loc_all))[0]

    weight = np.exp(-(tp_dis * tp_dis) / (7000 * 7000))
    for _, w in enumerate(weight):
        phi[_] = w * phi[_]

    s = candidate_set['t_p_i'].searchsorted(i, side='left')
    e = candidate_set['t_p_i'].searchsorted(i, side='right')

    c_s_f_v = []
    c_s_p = []
    for k in range(e - s):
        c_s = candidate_set.copy()
        f_value, P = find_sequence(c_s, weight, phi, i, k, e - s, trajectory_len)
        c_s_f_v.append(f_value)
        if np.isinf(f_value):
            # print('Bad local optimal path occurs, ignored.')
            continue
        c_s_p.append(P)
    return c_s_f_v, c_s_p


def vote(res_set, seg_name):
    vote = [dict() for idx in range(len(res_set))]
    for idx, item in enumerate(res_set):
        for path in item[1]:
            for i_idx, c_p in enumerate(path):
                vote[i_idx][c_p] = vote[i_idx][c_p] + 1 if c_p in vote[i_idx] else 1
    global_optimal_path = []
    for idx, item in enumerate(vote):
        best = []
        best_v = 0
        for k, v in item.items():
            if v > best_v:
                best = [k]
                best_v = v
            elif v == best_v:
                best.append(k)
        if not len(best):
            return global_optimal_path

        global_optimal_path.append(best[np.argmax([res_set[idx][0][i] for i in best])])
        if np.isinf(best[np.argmax([res_set[idx][0][i] for i in best])]):
            print(seg_name, "卧槽，坏了")
    return global_optimal_path


if __name__ == '__main__':
    display.configure_pandas()

    ssm_path = 'result/ssm'
    diag_M = np.load(os.path.join(ssm_path, 'temp.npy'))

    # load trajectory data
    data = data_loader.load_vehicles(n=1, max_length=3)
    cur_vehicle = data[0][0]

    # load candidate set
    candidate_set = pd.read_csv('result/candidate_point.csv')

    first_seg_len = candidate_set.loc[candidate_set['t_p_i']==candidate_set.index[0]].shape[0]
    M = []
    for t_p_i in range(candidate_set['t_p_i'].min(), candidate_set['t_p_i'].max()):
        pre_s = candidate_set['t_p_i'].searchsorted(t_p_i, side='left')
        pre_e = candidate_set['t_p_i'].searchsorted(t_p_i, side='right')
        next_s = candidate_set['t_p_i'].searchsorted(t_p_i + 1, side='left')
        next_e = candidate_set['t_p_i'].searchsorted(t_p_i + 1, side='right')
        M.append(diag_M[pre_s:pre_e, next_s - first_seg_len:next_e - first_seg_len])

    epsilon_u = 5
    epsilon_sigma = 10
    candidate_set['epsilon'] = (
        norm(epsilon_u, epsilon_sigma).pdf(candidate_set['residual'])
        * epsilon_sigma
        * math.sqrt(2 * math.pi)
    )
    print(candidate_set)
    c_s = candidate_set.groupby('t_p_i').apply(lambda x: x.reset_index(drop=True))
    c_s.index.rename(['i', 'k'], inplace=True)
    c_s.drop(columns='t_p_i', inplace=True)

    F = []
    P = []
    res_set = []
    client = Client(n_workers=4)
    for i in range(candidate_set['t_p_i'].min(), candidate_set['t_p_i'].max() + 1):
        res = delayed(traverse_trajectory_point)(i, M, cur_vehicle, candidate_set, c_s, 10)
        res_set.append(res)
    compute = delayed(vote)(res_set, F, P)
    global_optimal_path = compute.compute()
    print(c_s.loc[[(i, j) for i, j in enumerate(global_optimal_path)]])
    client.close()
