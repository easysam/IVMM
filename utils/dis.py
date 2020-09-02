"""Functions to calculate distances and find nearest node/edge(s) to point(s)."""

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

from osmnx import utils
from osmnx import utils_geo
from osmnx import utils_graph

# scipy and sklearn are optional dependencies for faster nearest node search
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
try:
    from sklearn.neighbors import BallTree
except ImportError:
    BallTree = None


def get_nearest_edge(G, point, return_geom=False, return_dist=False):
    """
    Return the nearest edge to a point, by minimum euclidean distance.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    point : tuple
        the (lat, lng) or (y, x) point for which we will find the nearest edge
        in the graph
    return_geom : bool
        Optionally return the geometry of the nearest edge
    return_dist : bool
        Optionally return the distance in graph's coordinates' units between
        the point and the nearest edge

    Returns
    -------
    tuple
        Graph edge unique identifier as a tuple of (u, v, key).
        Or a tuple of (u, v, key, geom) if return_geom is True.
        Or a tuple of (u, v, key, dist) if return_dist is True.
        Or a tuple of (u, v, key, geom, dist) if return_geom and return_dist are True.
    """
    # get u, v, key, geom from all the graph edges
    gdf_edges = utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    edges = gdf_edges[["u", "v", "key", "geometry"]].values

    # convert lat/lng point to x/y for shapely distance operation
    xy_point = Point(reversed(point))

    # calculate euclidean distance from each edge's geometry to this point
    edge_distances = [(edge, xy_point.distance(edge[3])) for edge in edges]

    # the nearest edge minimizes the distance to the point
    (u, v, key, geom), dist = min(edge_distances, key=lambda x: x[1])
    utils.log(f"Found nearest edge ({u, v, key}) to point {point}")

    # return results requested by caller
    if return_dist and return_geom:
        return u, v, key, geom, dist
    elif return_dist:
        return u, v, key, dist
    elif return_geom:
        return u, v, key, geom
    else:
        return u, v, key


def get_nearest_edges(G, X, Y, method=None, dist=0.0001):
    """
    Return the graph edges nearest to a list of points.

    Pass in points as separate vectors of X and Y coordinates. The 'kdtree'
    method is by far the fastest with large data sets, but only finds
    approximate nearest edges if working in unprojected coordinates like
    lat-lng (it precisely finds the nearest edge if working in projected
    coordinates). The 'balltree' method is second fastest with large data
    sets, but it is precise if working in unprojected coordinates like
    lat-lng. As a rule of thumb, if you have a small graph just use
    method=None. If you have a large graph with lat-lng coordinates, use
    method='balltree'. If you have a large graph with projected coordinates,
    use method='kdtree'. Note that if you are working in units of lat-lng,
    the X vector corresponds to longitude and the Y vector corresponds
    to latitude. The method creates equally distanced points along the edges
    of the network. Then, these points are used in a kdTree or BallTree search
    to identify which is nearest.Note that this method will not give the exact
    perpendicular point along the edge, but the smaller the *dist* parameter,
    the closer the solution will be.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    X : list-like
        The vector of longitudes or x's for which we will find the nearest
        edge in the graph. For projected graphs, use the projected coordinates,
        usually in meters.
    Y : list-like
        The vector of latitudes or y's for which we will find the nearest
        edge in the graph. For projected graphs, use the projected coordinates,
        usually in meters.
    method : string {None, 'kdtree', 'balltree'}
        Which method to use for finding nearest edge to each point.
        If None, we manually find each edge one at a time using
        get_nearest_edge. If 'kdtree' we use
        scipy.spatial.cKDTree for very fast euclidean search. Recommended for
        projected graphs. If 'balltree', we use sklearn.neighbors.BallTree for
        fast haversine search. Recommended for unprojected graphs.

    dist : float
        spacing length along edges. Units are the same as the geom; Degrees for
        unprojected geometries and meters for projected geometries. The smaller
        the value, the more points are created.

    Returns
    -------
    ne : np.array
        array of nearest edges represented by u and v (the IDs of the nodes
        they link) and key
    """
    if method is None:
        # calculate nearest edge one at a time for each (y, x) point
        ne = [get_nearest_edge(G, (y, x)) for x, y in tqdm(zip(X, Y))]

    elif method == "kdtree":

        # check if we were able to import scipy.spatial.cKDTree successfully
        if not cKDTree:
            raise ImportError("The scipy package must be installed to use this optional feature.")

        # transform graph into DataFrame
        edges = utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

        # transform edges into evenly spaced points
        edges["points"] = edges.apply(
            lambda x: utils_geo.redistribute_vertices(x.geometry, dist), axis=1
        )

        # develop edges data for each created points
        extended = (
            edges["points"]
            .apply([pd.Series])
            .stack()
            .reset_index(level=1, drop=True)
            .join(edges)
            .reset_index()
        )

        # Prepare btree arrays
        nbdata = np.array(
            list(
                zip(
                    extended["Series"].apply(lambda x: x.x), extended["Series"].apply(lambda x: x.y)
                )
            )
        )

        # build a k-d tree for euclidean nearest node search
        btree = cKDTree(data=nbdata, compact_nodes=True, balanced_tree=True)

        # query the tree for nearest node to each point
        points = np.array([X, Y]).T
        dist, idx = btree.query(points, k=1)  # Returns ids of closest point
        eidx = extended.loc[idx, "index"]
        ne = edges.loc[eidx, ["u", "v", "key"]]

    elif method == "balltree":
        # check if we were able to import sklearn.neighbors.BallTree successfully
        if not BallTree:
            raise ImportError(
                "The scikit-learn package must be installed to use this optional feature."
            )

        # transform graph into DataFrame
        edges = utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

        # transform edges into evenly spaced points
        edges["points"] = edges.apply(
            lambda x: utils_geo.redistribute_vertices(x.geometry, dist), axis=1
        )

        # develop edges data for each created points
        extended = (
            edges["points"]
            .apply([pd.Series])
            .stack()
            .reset_index(level=1, drop=True)
            .join(edges)
            .reset_index()
        )

        # haversine requires data in form of [lat, lng] and inputs/outputs in units of radians
        nodes = pd.DataFrame(
            {
                "x": extended["Series"].apply(lambda x: x.x),
                "y": extended["Series"].apply(lambda x: x.y),
            }
        )
        nodes_rad = np.deg2rad(nodes[["y", "x"]].values.astype(np.float))
        points = np.array([Y, X]).T
        points_rad = np.deg2rad(points)

        # build a ball tree for haversine nearest node search
        tree = BallTree(nodes_rad, metric="haversine")

        # query the tree for nearest node to each point
        idx = tree.query(points_rad, k=5, return_distance=False)
        print(idx)
        eidx = extended.loc[idx[:, 0], "index"]
        ne = edges.loc[eidx, ["u", "v", "key"]]

    else:
        raise ValueError("You must pass a valid method name, or None.")

    utils.log(f"Found nearest edges to {len(X)} points")

    return np.array(ne)
