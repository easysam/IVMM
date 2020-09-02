import osmnx as ox
import pandas as pd
import networkx as nx
import numpy as np

from utils import data_loader, display


def compute_shortest_path_len(road_network, pre_point, next_point):
    shortest_path_length = np.PINF
    node_path = []
    if (pre_point['e_i'] == next_point['e_i']) \
            and not ((pre_point['oneway'])
                     and (pre_point['i_p_i'] != next_point['i_p_i'])
                     and (next_point['edge_progress'] - pre_point['edge_progress'] < 0)):
        shortest_path_length = abs(next_point['edge_progress'] - pre_point['edge_progress']) * pre_point['length']
        print(pre_point['u'], pre_point['v'], pre_point['end_node'], next_point['end_node'])
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
                    node_path = [pre_point['v'], next_point['u']]
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
                    node_path = [pre_point['v'], next_point['u']]
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
                    node_path = [pre_point['u'], next_point['u']]
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
                    node_path = [pre_point['v'], next_point['u']]
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
                    node_path = [pre_point['v'], next_point['v']]
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
                    node_path = [pre_point['v'], next_point['u']]
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
                    node_path = [pre_point['u'], next_point['u']]
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
                    node_path = [pre_point['v'], next_point['v']]
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
                    node_path = [pre_point['u'], next_point['v']]
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
                    node_path = [pre_point[pre_node], next_point['u']]
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
                    node_path = [pre_point[pre_node], next_point['u']]
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
                    node_path = [pre_point[pre_node], next_point['v']]
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
                    node_path = [pre_point['v'], next_point[next_node]]
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
                    node_path = [pre_point['v'], next_point[next_node]]
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
                    node_path = [pre_point['u'], next_point[next_node]]
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
                node_path = [pre_point[pre_node], next_point[next_node]]
        except:
            pass

    if len(node_path):
        node_path = nx.shortest_path(road_network, node_path[0], node_path[1], weight='length')
    return shortest_path_length, node_path


if __name__ == '__main__':
    display.configure_pandas()
    # road_network = data_loader.load_drive_graph()
    # ox.io.save_graphml(road_network, filepath='db/shenzhen-drive.osm', gephi=False, encoding='utf-8')
    ## the road network data is created by above code segment
    road_network = ox.load_graphml('db/shenzhen-drive-20200813.osm')

    path = 'result/seg/0-1/粤B14337_4832.csv'
    road_path = pd.read_csv(path)
    shenzhen_road_network = ox.truncate.truncate_graph_bbox(road_network,
                                                            road_path['y'].max() + 0.03,
                                                            road_path['y'].min() - 0.03,
                                                            road_path['x'].max() + 0.03,
                                                            road_path['x'].min() - 0.03)

    route = []
    for i in range(len(road_path.index))[1:]:
        if 75 == i:
            print('cool')
        _, sub_path = compute_shortest_path_len(shenzhen_road_network, road_path.iloc[i-1], road_path.iloc[i])
        print(sub_path)
        if len(route) and len(sub_path) and (route[-1] == sub_path[0]):
            route.extend(sub_path[1:])
        else:
            route.extend(sub_path)
    print(route)

    ox.plot_graph_route(shenzhen_road_network, route, route_linewidth=2, dpi=1200)
