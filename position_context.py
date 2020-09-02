import os
import math
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import scipy

from pathlib import Path
from scipy.stats import norm

from utils import data_loader
import utils.vector_haversine_distances as v_harv_dis


def compute_epsilon(residual, epsilon_u=5, epsilon_sigma=10):
    return (
            norm(epsilon_u, epsilon_sigma).pdf(residual)
            * epsilon_sigma
            * math.sqrt(2 * math.pi)
    )


def compute_shortest_path_len(road_network, pre_point, next_point):
    shortest_path_length = np.PINF
    if (pre_point['e_i'] == next_point['e_i']) \
            and not ((pre_point['oneway'])
                     and (pre_point['i_p_i'] != next_point['i_p_i'])
                     and (next_point['edge_progress'] - pre_point['edge_progress'] < 0)):
        shortest_path_length = abs(next_point['edge_progress'] - pre_point['edge_progress']) * pre_point['length']
    elif (0 == pre_point['end_node']) & (0 == next_point['end_node']):
        if pre_point['oneway'] & next_point['oneway']:
            
            try:
                temp_length = (pre_point['length'] * (1 - pre_point['edge_progress'])
                               + nx.shortest_path_length(road_network,
                                                         pre_point['v'], next_point['u'],
                                                         weight='length')
                               + next_point['length'] * next_point['edge_progress']
                               )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
        elif ~pre_point['oneway'] & next_point['oneway']:
            # pre seg forward, next seg forward
            try:
                temp_length = (pre_point['length'] * (1 - pre_point['edge_progress'])
                               + nx.shortest_path_length(road_network,
                                                         pre_point['v'], next_point['u'],
                                                         weight='length')
                               + next_point['length'] * next_point['edge_progress']
                               )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
            # pre seg backward, next seg forward
            try:
                temp_length = (pre_point['length'] * pre_point['edge_progress']
                               + nx.shortest_path_length(road_network,
                                                         pre_point['u'], next_point['u'],
                                                         weight='length')
                               + next_point['length'] * next_point['edge_progress']
                               )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
        elif pre_point['oneway'] & ~next_point['oneway']:
            # pre seg forward, next seg forward
            try:
                temp_length = (pre_point['length'] * (1 - pre_point['edge_progress'])
                               + nx.shortest_path_length(road_network,
                                                         pre_point['v'], next_point['u'],
                                                         weight='length')
                               + next_point['length'] * next_point['edge_progress']
                               )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
            # pre seg forward, next seg backward
            try:
                temp_length = (pre_point['length'] * (1 - pre_point['edge_progress'])
                               + nx.shortest_path_length(road_network,
                                                         pre_point['v'], next_point['v'],
                                                         weight='length')
                               + next_point['length'] * (1 - next_point['edge_progress'])
                               )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
        elif ~pre_point['oneway'] & ~next_point['oneway']:
            # pre seg forward, next seg forward
            try:
                temp_length = (pre_point['length'] * (1 - pre_point['edge_progress'])
                               + nx.shortest_path_length(road_network,
                                                         pre_point['v'], next_point['u'],
                                                         weight='length')
                               + next_point['length'] * next_point['edge_progress']
                               )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
            # pre seg backward, next seg forward
            try:
                temp_length = (pre_point['length'] * pre_point['edge_progress']
                               + nx.shortest_path_length(road_network,
                                                         pre_point['u'], next_point['u'],
                                                         weight='length')
                               + next_point['length'] * next_point['edge_progress']
                               )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
            # pre seg forward, next seg backward
            try:
                temp_length = (
                        pre_point['length'] * (1 - pre_point['edge_progress'])
                        + nx.shortest_path_length(road_network, pre_point['v'], next_point['v'], weight='length')
                        + next_point['length'] * (1 - next_point['edge_progress'])
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
            # pre seg backward, next seg backward
            try:
                temp_length = (
                        pre_point['length'] * pre_point['edge_progress']
                        + nx.shortest_path_length(road_network, pre_point['u'], next_point['v'], weight='length')
                        + next_point['length'] * (1 - next_point['edge_progress'])
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
    elif (0 != pre_point['end_node']) & (0 == next_point['end_node']):
        if 1 == pre_point['end_node']:
            pre_node = 'u'
        elif 2 == pre_point['end_node']:
            pre_node = 'v'
        if next_point['oneway']:
            try:
                temp_length = (
                        nx.shortest_path_length(road_network, pre_point[pre_node], next_point['u'],
                                                weight='length')
                        + next_point['length'] * next_point['edge_progress']
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
        elif ~next_point['oneway']:
            # next seg forward
            try:
                temp_length = (
                        nx.shortest_path_length(road_network, pre_point[pre_node], next_point['u'],
                                                weight='length')
                        + next_point['length'] * next_point['edge_progress']
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
            # next seg backward
            try:
                temp_length = (
                        nx.shortest_path_length(road_network, pre_point[pre_node], next_point['v'],
                                                weight='length')
                        + next_point['length'] * (1 - next_point['edge_progress'])
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
    elif (0 == pre_point['end_node']) & (0 != next_point['end_node']):
        if 1 == next_point['end_node']:
            next_node = 'u'
        elif 2 == next_point['end_node']:
            next_node = 'v'
        if pre_point['oneway']:
            try:
                temp_length = (
                        pre_point['length'] * (1 - pre_point['edge_progress'])
                        + nx.shortest_path_length(road_network, pre_point['v'], next_point[next_node],
                                                  weight='length')
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
        elif ~pre_point['oneway']:
            # next seg forward
            try:
                temp_length = (
                        pre_point['length'] * (1 - pre_point['edge_progress'])
                        + nx.shortest_path_length(road_network, pre_point['v'], next_point[next_node],
                                                  weight='length')
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
            # next seg backward
            try:
                temp_length = (
                        pre_point['length'] * pre_point['edge_progress']
                        + nx.shortest_path_length(road_network, pre_point['u'], next_point[next_node],
                                                  weight='length')
                )
                if temp_length < shortest_path_length:
                    shortest_path_length = temp_length
            except:
                pass
    elif (0 != pre_point['end_node']) & (0 != next_point['end_node']):
        if 1 == pre_point['end_node']:
            pre_node = 'u'
        elif 2 == pre_point['end_node']:
            pre_node = 'v'
        if 1 == next_point['end_node']:
            next_node = 'u'
        elif 2 == next_point['end_node']:
            next_node = 'v'
        try:
            temp_length = (
                nx.shortest_path_length(road_network, pre_point[pre_node], next_point[next_node],
                                        weight='length')
            )
            if temp_length < shortest_path_length:
                shortest_path_length = temp_length
        except:
            pass
    return shortest_path_length


def compute_static_score_matrix(candidate_set, trajectory, road_network, traj_len):
    M = []
    for t_p_i in range(traj_len - 1):
        pre_s = candidate_set['t_p_i'].searchsorted(t_p_i, side='left')
        pre_e = candidate_set['t_p_i'].searchsorted(t_p_i, side='right')
        next_s = candidate_set['t_p_i'].searchsorted(t_p_i + 1, side='left')
        next_e = candidate_set['t_p_i'].searchsorted(t_p_i + 1, side='right')
        M_i = np.full((pre_e - pre_s, next_e - next_s), np.NINF)
        # print(t_p_i)
        for i in range(pre_e - pre_s):
            for j in range(next_e - next_s):
                epsilon = candidate_set.iat[next_s + j, candidate_set.columns.get_loc('epsilon')]
                d_from_pre = trajectory.iloc[t_p_i + 1, trajectory.columns.get_loc('dis_f_pre')]
                pre_seg = candidate_set.iloc[pre_s + i]
                next_seg = candidate_set.iloc[next_s + j]
                shortest_path_length = compute_shortest_path_len(road_network, pre_seg, next_seg)
                if shortest_path_length <= d_from_pre:
                    M_i[i, j] = epsilon
                elif np.isinf(shortest_path_length):
                    M_i[i, j] = np.NINF
                else:
                    # print(epsilon, d_from_pre, shortest_path_length, epsilon * d_from_pre / shortest_path_length)
                    M_i[i, j] = epsilon * d_from_pre / shortest_path_length
        # print(M_i)
        M.append(M_i)
    return M


if __name__ == '__main__':
    # load road network
    truncated_roads = ox.load_graphml('db/truncated_graph.graphml')
    edges = ox.utils_graph.graph_to_gdfs(truncated_roads, nodes=False,
                                         fill_edge_geometry=True)

    # load candidate set
    candidate_set = pd.read_csv('result/candidate_point.csv')
    # load trajectory data
    data = data_loader.load_vehicles(n=1, max_length=3)
    cur_vehicle = data[0]

    cur_vehicle['dis_f_pre'] = v_harv_dis.haversine_np(cur_vehicle['longitude'],
                                                          cur_vehicle['latitude'],
                                                          cur_vehicle.shift()['longitude'],
                                                          cur_vehicle.shift()['latitude'])

    # compute epsilon of each candidate point
    candidate_set['epsilon'] = compute_epsilon(candidate_set['residual'])

    # develop edges data(oneway, length, u, v) for each candidate points
    candidate_set = candidate_set.merge(edges[['oneway', 'length', 'u', 'v']], left_on='e_i', right_index=True)

    M = compute_static_score_matrix(candidate_set, cur_vehicle, truncated_roads)

    M = scipy.linalg.block_diag(*M)

    # static score matrix path
    ssm_path = 'result/ssm'
    Path(ssm_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(ssm_path, 'temp'), M)

