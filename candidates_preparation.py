from tqdm import tqdm
import osmnx as ox
from utils import data_loader
import utils.display as display

import pandas as pd
import numpy as np
import geopandas as gpd
from osmnx import utils_graph
from osmnx import utils_geo
try:
    from sklearn.neighbors import BallTree
except ImportError:
    BallTree = None


def extended_edges(edges):
    # transform edges into evenly spaced points
    edges["points"] = edges.apply(
        lambda x: utils_geo.redistribute_vertices(x.geometry, dist=0.0001), axis=1
    )

    # develop edges data for each created points
    extended = (
        edges["points"]
            .apply([pd.Series])
            .stack()
            .reset_index(level=1)
            .join(edges.drop(columns=['oneway', 'length', 'geometry',
                                      'u', 'v', 'points']))
            .reset_index()
    )

    # give each created points [index of edge 'e_i'] and [index of point in edge 'p_i_e_i']
    extended.rename(columns={'index': 'e_i', 'level_1': 'p_i_e_i'}, inplace=True)
    extended['end_node'] = 0
    extended.loc[extended['e_i'] != extended['e_i'].shift(), 'end_node'] = 1
    extended.loc[extended['e_i'] != extended['e_i'].shift(-1), 'end_node'] = 2
    points_count = extended.groupby(['e_i']).size()
    # 'edge_progress' means distance progress along edge
    points_count.rename('edge_points_count', inplace=True)
    extended = extended.merge(points_count, left_on='e_i', right_index=True, )
    extended['edge_progress'] = extended['p_i_e_i'] / (extended['edge_points_count'] - 1)
    return extended


def make_tree(gdf_extended_edges):
    # check if we were able to import sklearn.neighbors.BallTree successfully
    if not BallTree:
        raise ImportError(
            "The scikit-learn package must be installed to use this optional feature."
        )

    # haversine requires data in form of [lat, lng] and inputs/outputs in units of radians
    gpd_extend = gpd.GeoDataFrame(gdf_extended_edges, geometry='Series')
    nodes = pd.DataFrame({"x": gpd_extend['Series'].x, "y": gpd_extend['Series'].y})
    nodes_rad = np.deg2rad(nodes[["y", "x"]].values.astype(np.float))

    # build a ball tree for haversine nearest node search
    tree = BallTree(nodes_rad, metric="haversine")
    return tree


def dis_bt_tp_ep(trajectory_point, ball_tree):
    # distance between trajectory points and extended points

    # prepare points to be queried
    X = trajectory_point['longitude']
    Y = trajectory_point['latitude']
    points = np.array([Y, X]).T
    points_rad = np.deg2rad(points)

    # query the tree for node in a radius to each trajectory point
    r = 150 / 6378000
    idx, dis = ball_tree.query_radius(points_rad, r=r, return_distance=True)

    # special case: all trajectory points have no neighbor points
    if 0 == sum([len(sub_idx) for sub_idx in idx]):
        return None

    # associate distances to edges
    # build dis between trajectory points and each neighbor points
    dis_b_tp_pie = (
            pd.Series(dis)
            .apply([pd.Series])
            .stack()
            * 6371000
    )
    # add global interpolated index to each neighbor points
    dis_b_tp_pie.rename(columns={'Series': 'residual'}, inplace=True)
    idx = (
        pd.Series(idx)
            .apply(pd.Series, dtype=int)
            .stack()
    )
    dis_b_tp_pie.set_index([dis_b_tp_pie.index.get_level_values(0), idx], inplace=True)

    # rename index to trajectory points index and (global, not inside edge)interpolated points index
    dis_b_tp_pie.index.rename(['t_p_i', 'i_p_i'], inplace=True)

    # res.sort_values('residual', inplace=True)
    return dis_b_tp_pie


def make_candidates_set(dis_b_tp_pie, extended):
    dis_b_tp_pie = dis_b_tp_pie.merge(extended[['e_i', 'end_node', 'edge_progress']],
                                      left_on='i_p_i',
                                      right_index=True,
                                      how='left')

    # distance between trajectory points and edges
    dis_b_tp_e_i = dis_b_tp_pie.groupby(['t_p_i', 'e_i'])['residual'].idxmin()
    dis_b_tp_e = dis_b_tp_pie.loc[dis_b_tp_e_i]

    # 5 nearest edge is candidate edge
    candidate_set = (
        dis_b_tp_e
            .groupby(level=0, group_keys=False)
            .apply(lambda grp: grp.nsmallest(5, 'residual'))
    )

    gpd_extend = gpd.GeoDataFrame(extended, geometry='Series')
    nodes = pd.DataFrame({"x": gpd_extend['Series'].x, "y": gpd_extend['Series'].y})
    candidate_set = candidate_set.merge(nodes, left_on='i_p_i', right_index=True, how='left')
    candidate_set.reset_index(inplace=True)
    candidate_set = candidate_set.round({'x': 6, 'y': 6})
    candidate_set.drop_duplicates(subset=['t_p_i', 'x', 'y'], inplace=True)
    return candidate_set


if __name__ == '__main__':
    display.configure_pandas()
    tqdm.pandas()

    # load trajectory data
    data = data_loader.load_vehicles(n=1, max_length=0)
    cur_vehicle = data[0][0]

    # load road network
    truncated_roads = ox.load_graphml('db/truncated_graph.graphml')
    # ox.plot_graph(truncated_roads)
    # truncated_roads = data_loader.load_drive_graph()

    # transform graph into DataFrame
    edges = utils_graph.graph_to_gdfs(truncated_roads, nodes=False, fill_edge_geometry=True)
    edges.drop(columns=['osmid', 'highway', 'bridge', 'name', 'key'], inplace=True)

    # extended edges to points, for following tree
    extended = extended_edges(edges)

    tree = make_tree(extended)

    dis_b_tp_pie = dis_bt_tp_ep(cur_vehicle, tree)

    candidates_set = make_candidates_set(dis_b_tp_pie, extended)

    print(candidates_set)
    extended.to_csv('result/extended_edges.csv', index=False)
    candidates_set.to_csv('result/candidate_point.csv', index=False)
