import os
import datetime
import pandas as pd
import utils.vector_haversine_distances as vector_haversine_distances

from pathlib import Path


def datetime_format_transfer(df, columns_list, format_str='%Y-%m-%dT%H:%M:%S.%fZ'):
    for column in columns_list:
        df[column] = df[column].apply(lambda x: datetime.datetime.strptime(x, format_str))
    return df


def split_trajectory_by_rest_status(trajectory, destination_path="db/seg",
                                    plate="plate", speed='velocity',
                                    timestamp="timestamp", longitude="longitude", latitude="latitude",
                                    condition=None, save=True):
    """
    split_trajectory_by_rest_status
    :param save:
    :param speed:
    :param trajectory:
    :param destination_path:
    :param plate:
    :param timestamp:
    :param longitude:
    :param latitude:
    :param condition: for shenzhen 2018 data, it claim 'use' is '营运'
    :return:
    """
    # if there is(are) condition(s) to filter trajectory, do it
    if not condition:
        mark = [True] * len(trajectory.index)
        for k, v in condition:
            mark = mark & trajectory[k] == v
        trajectory = trajectory.loc[mark]
    # Calculate distance from previous point

    trajectory['dis_f_pre'] = vector_haversine_distances.haversine_np(trajectory[longitude],
                                                                      trajectory[latitude],
                                                                      trajectory[longitude].shift(),
                                                                      trajectory[latitude].shift())

    # Calculate time interval length from previous point
    trajectory['interval'] = trajectory[timestamp] - trajectory[timestamp].shift()
    # Set dis_f_pre and interval of first point of each plate as None
    trajectory.loc[trajectory[plate] != trajectory[plate].shift(), ['dis_f_pre', 'interval']] = [None, None]
    # Give annotation of big interval point, which defined by condition the interval bigger than 30 minutes
    trajectory['big_interval'] = trajectory['interval'] > datetime.timedelta(minutes=30)
    # Give 'valid' annotation, which indicate points not caused by GPS data missing
    # data missing means: there is big interval, and the begin and end of the interval is far
    trajectory['valid'] = ~ (trajectory['big_interval'] & (trajectory['dis_f_pre'] > 0.5))
    # Give stop annotation, which indicate the vehicle is stay
    trajectory['stop'] = (((trajectory['dis_f_pre'] < 0.1) & ~trajectory['big_interval'])
                          | (trajectory['big_interval'] & trajectory['valid']))

    # group trajectory by continuous same stop status point
    trajectory['grp'] = ((trajectory['stop'] != trajectory['stop'].shift())
                         | (trajectory[plate] != trajectory[plate].shift())
                        ).cumsum()
    # Add last timestamp to each point, for later usage
    trajectory['last_timestamp'] = trajectory[timestamp].shift()
    trajectory.loc[trajectory[plate] != trajectory[plate].shift(), 'last_timestamp'] = None
    # Make a special trajectory that switch timestamp to last one if
    # two condition is satisfied (1) point is end of a big interval (2) point is stop.
    # Doing this to avoid missing the stay interval before the point which is in end of a big interval
    df_special_traj = trajectory.copy()
    df_special_traj['begin_time'] = df_special_traj[timestamp]
    df_special_traj.loc[(df_special_traj['big_interval']) & (df_special_traj['valid']), 'begin_time'] = \
    df_special_traj.loc[(df_special_traj['big_interval']) & (df_special_traj['valid']), 'last_timestamp']
    # Select rest group, which satisfied (1) group status is stop (2) group duration longer than 30 minutes
    rest_grp = (df_special_traj.groupby('grp')['stop'].first()
            & (
                    (df_special_traj.groupby('grp')[timestamp].last() -
                     df_special_traj.groupby('grp')['begin_time'].first()
                     ) > pd.Timedelta(minutes=30)
               )
            )
    df_special_traj.loc[df_special_traj['grp'].isin(rest_grp.loc[rest_grp==True].index), 'rest'] = True
    df_special_traj['rest'] = df_special_traj['rest'].fillna(value=False)
    # Grouping by continuous same rest status point
    df_special_traj['new_seg'] = 0
    df_special_traj.loc[((df_special_traj['rest'] != df_special_traj['rest'].shift())
                         | (df_special_traj[plate] != df_special_traj[plate].shift())), 'new_seg'] = 1
    df_special_traj['grp'] = df_special_traj['new_seg'].cumsum()
    if save:
        def save_non_rest(seg, path):
            if not seg.at[seg.index[0], 'rest']:
                seg[[plate, longitude, latitude, timestamp, speed, 'dis_f_pre']]\
                    .to_csv(os.path.join(path,
                                         seg.at[seg.index[0], plate] + '_' + str(seg.name) + '.csv'),
                            index=False)
        Path(destination_path).mkdir(parents=True, exist_ok=True)
        df_special_traj.groupby('grp').apply(save_non_rest, path=destination_path,)

