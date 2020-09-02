import os
import pickle
import osmnx as ox
import pandas as pd
import candidates_preparation
import position_context
import interactive_voting

from tqdm import tqdm
from utils import data_loader

tqdm.pandas()


def task(m=0, n=94739, res_path='result/seg'):
    data = data_loader.load_vehicles(m=m, n=n, max_length=0)
    for vehicle, seg_name in tqdm(data):
        if len(vehicle.index) < 2:
            print(seg_name, 'length less than 2')
            continue
        dis_b_tp_pie = candidates_preparation.dis_bt_tp_ep(vehicle, tree)
        if dis_b_tp_pie is None:
            print(seg_name, 'No candidate points.')
            continue
        candidate_set = candidates_preparation.make_candidates_set(dis_b_tp_pie, extended)

        # compute epsilon of each candidate point
        candidate_set['epsilon'] = position_context.compute_epsilon(candidate_set['residual'])
        # develop edges data(oneway, length, u, v) for each candidate points
        candidate_set = candidate_set.merge(edges[['oneway', 'length', 'u', 'v']], left_on='e_i', right_index=True,
                                            how='left')
        vehicle = vehicle.loc[vehicle.index.isin(candidate_set['t_p_i'].unique())]
        # reset trajectory points index to order number from 0
        vehicle.reset_index(drop=True, inplace=True)
        candidate_set['t_p_i'] = (candidate_set['t_p_i'] != candidate_set['t_p_i'].shift()).cumsum() - 1

        trajectory_len = len(vehicle.index)

        # print(candidate_set.iloc[295:310])
        # %%

        M = position_context.compute_static_score_matrix(candidate_set, vehicle, road_network, trajectory_len)

        # %%

        c_s = candidate_set.groupby('t_p_i').apply(lambda x: x.reset_index(drop=True))
        c_s.index.rename(['i', 'k'], inplace=True)
        c_s.drop(columns='t_p_i', inplace=True)

        # %%

        res_set = []
        for i in range(trajectory_len):
            # res = delayed(interactive_voting.traverse_trajectory_point)(i, M, vehicle, candidate_set, c_s)
            res = interactive_voting.traverse_trajectory_point(i, M, vehicle, candidate_set, trajectory_len)
            res_set.append(res)
        # compute = delayed(interactive_voting.vote)(res_set)
        # global_optimal_path = compute.compute()
        global_optimal_path = interactive_voting.vote(res_set, seg_name)
        # P = c_s.loc[[(i, j) for i, j in enumerate(global_optimal_path)]]
        if not len(global_optimal_path):
            print(seg_name, 'no connective path')
            continue
        P = candidate_set.iloc[[candidate_set['t_p_i'].searchsorted(i) + j for i, j in enumerate(global_optimal_path)]]
        P.reset_index(drop=True, inplace=True)
        pd.concat([P[['i_p_i', 'e_i', 'end_node', 'edge_progress', 'x', 'y', 'oneway', 'length', 'u', 'v']],
                   vehicle], axis=1).to_csv(os.path.join(res_path, seg_name))
    # client.close()


if __name__ == '__main__':
    # road_network = data_loader.load_drive_graph()
    # ox.io.save_graphml(road_network, filepath='db/shenzhen-drive.osm', gephi=False, encoding='utf-8')
    ## or
    # road_network = ox.graph_from_place('Shenzhen, Guangdong, China', network_type='drive')
    ## the road network data is created by above code segment
    road_network = ox.load_graphml('db/shenzhen-drive-20200813.osm')

    edges = ox.utils_graph.graph_to_gdfs(road_network, nodes=False, fill_edge_geometry=True)
    edges.drop(columns=['osmid', 'highway', 'bridge', 'name', 'key'], inplace=True)

    # extended = candidates_preparation.extended_edges(edges)
    # with open('db/shenzhen_drive_edges_extend.pkl', 'wb') as f:
    #     pickle.dump(extended, f)
    ## extended edges is created by above code segment
    with open('db/shenzhen_drive_edges_extend.pkl', 'rb') as f:
        extended = pickle.load(f)

    # tree = candidates_preparation.make_tree(extended)
    # with open('db/shenzhen_drive_extended_edges_tree.pkl', 'wb') as f:
    #     pickle.dump(tree, f)
    ## tree is created by above code segment
    with open('db/shenzhen_drive_extended_edges_tree.pkl', 'rb') as f:
        tree = pickle.load(f)

    task(m=10625, n=20000, res_path='result/seg/1-2')
