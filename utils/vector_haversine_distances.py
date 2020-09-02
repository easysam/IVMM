import numpy as np
import math


def haversine_np(lon1, lat1, lon2, lat2, miles=False):
    """
    Calculate the great-circle distance bewteen two points on the Earth surface.

    :input: 4 GPS coordinates, containing the latitude and longitude of each point
    in decimal degrees.

    Example: haversine(45.7597, 4.8422, 48.8567, 2.3508)

    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.

    """
    AVG_EARTH_RADIUS = 6371.0088  # in km

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    d = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h * 1000  # in meters


if __name__ == '__main__':
    print('bottom left and bottom right: ', haversine_np(113.764635, 22.454727, 114.608972, 22.454727))
    # distance between bottom left and bottom right is: 86.76759310492864 KM
    print('top left and top right: ', haversine_np(113.764635, 22.842654, 114.608972, 22.842654))
    # distance between top left and top right is: 86.5228084907369 KM
    # difference is 224 M

    print('top left and bottom left: ', haversine_np(113.764635, 22.454727, 113.764635, 22.842654))
    # distance between top left and bottom left is 43.13551430844383 KM

    print(haversine_np([np.nan, 113.961098], [np.nan, 22.553101], [113.962997, 113.962303], [22.547001, 22.547001]))
    print(haversine_np(114.007401,22.535500, 114.0090009,22.53423323))
    print(haversine_np(114.0090009, 22.53423323,  114.042, 22.5312))