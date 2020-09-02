import pandas as pd
import osmnx as ox
import glob
import ntpath


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def load_vehicles(m=0, n=94739, max_length=0):
    vehicles = glob.glob('db/segment/*.csv')
    if n > len(vehicles):
        print('data_loader.load_vehicles: there is not existing %d vehicles, only %d' % (n, len(vehicles)))
    used = vehicles[m:n]
    res = []
    for vehicle in used:
        res.append((
            pd.read_csv(vehicle, usecols=['plate','longitude','latitude','timestamp','velocity','dis_f_pre'],
                        parse_dates=['timestamp']), path_leaf(vehicle)
        ))
    if max_length:
        for i in range(n):
            res[i] = (res[i][0].iloc[:max_length], res[i][1])
    return res


def load_drive_graph(path='db/shenzhen20170701-all.osm',):
    """
    load original open street map data of a city and extract drivable road network
    :param path: original osm xml format data
    :return: drivable graph
    """
    shenzhen_all = ox.graph_from_xml(path)
    sz_nodes, sz_edges = ox.graph_to_gdfs(shenzhen_all, fill_edge_geometry=True)

    not_valid_highway = 'cycleway|footway|path|pedestrian|steps|track|corridor|elevator' \
                        '|escalator|proposed|construction|bridleway|abandoned|platform' \
                        '|raceway|service'.split('|')
    not_valid_service = 'parking|parking_aisle|driveway|private|emergency_access'.split('|')
    print('original_edges', sz_edges.shape)

    sz_gdfs = sz_edges.loc[((sz_edges['highway'].notna())
                            & (sz_edges['area'] != 'yes')
                            & (sz_edges['access'] != 'private')
                            & (~sz_edges['service'].isin(not_valid_service)))]

    for tag in not_valid_highway:
        sz_gdfs = sz_gdfs.loc[sz_gdfs['highway'] != tag].copy(deep=True)
    print('drivable_edges:', sz_gdfs.shape)

    shenzhen_drive = ox.graph_from_gdfs(sz_nodes, sz_gdfs)

    return shenzhen_drive
