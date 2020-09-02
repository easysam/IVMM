import os
import tqdm
import datetime
import pandas as pd

from pathlib import Path
from utils import display
from utils import vector_haversine_distances

if __name__ == '__main__':
    tqdm.tqdm.pandas()
    display.configure_pandas()
    ###################################################################################################################
    # Function3: Split trajectory of each vehicle to individual files.
    # Create Time: 7.24 later
    # path = 'db/original_trajectories.csv'
    # trajectory = pd.read_csv(path, parse_dates=['timestamp'])
    #
    # path = 'db/trajectory_by_license/'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #
    # def split_by_vehicle(license_trajectory):
    #     license_trajectory.to_csv(path + license_trajectory.name + '.csv', index=False)
    #
    # trajectory.groupby('plate').progress_apply(split_by_vehicle)
    ###################################################################################################################

    ###################################################################################################################
    # Split trajectory of a vehicle to split files by stay point.
    # Create Time: 8.10
    path = 'db/original_trajectories.csv'
    trajectory = pd.read_csv(path, parse_dates=['timestamp'])

    trajectory['dis_f_pre'] = vector_haversine_distances.haversine_np(trajectory['longitude'],
                                                                         trajectory['latitude'],
                                                                         trajectory['longitude'].shift(),
                                                                         trajectory['latitude'].shift())
    trajectory['interval'] = trajectory['timestamp'] - trajectory['timestamp'].shift()
    print(trajectory.loc[(trajectory['plate'] == '粤B4BX08')
                         & (trajectory['timestamp'] > datetime.datetime(2014, 7, 16, 11, 40))])
    trajectory.loc[trajectory['plate'] != trajectory['plate'].shift(), ['dis_f_pre', 'interval']] = [None, None]
    print(trajectory.loc[(trajectory['plate'] == '粤B4BX08')
                         & (trajectory['timestamp'] > datetime.datetime(2014, 7, 16, 11, 40))])

    # 大间隔点
    trajectory['big_interval'] = trajectory['interval'] > datetime.timedelta(minutes=30)
    # 大间隔且点距超过0.1KM的定为异常点
    trajectory['valid'] = ~ (trajectory['big_interval'] & (trajectory['dis_f_pre'] > 0.5))
    # 给出停止点：速度等于0或者距离小于100m
    trajectory['stop'] = (((trajectory['velocity'] == 0) & ~trajectory['big_interval'])
                             | (trajectory['big_interval'] & trajectory['valid']))
    # 给出分组
    trajectory['grp'] = ((trajectory['stop'] != trajectory['stop'].shift())
                         | (trajectory['plate'] != trajectory['plate'].shift())
                         ).cumsum()

    # Add last timestamp to each point
    trajectory['last_timestamp'] = trajectory['timestamp'].shift()
    trajectory.loc[trajectory['plate'] != trajectory['plate'].shift(), 'last_timestamp'] = None

    # Make a special trajectory that switch timestamp to last one if point is end of a big interval.
    df_special_traj = trajectory.copy()
    df_special_traj['begin_time'] = df_special_traj['timestamp']
    df_special_traj.loc[(df_special_traj['big_interval']) & (df_special_traj['valid']), 'begin_time'] = \
    df_special_traj.loc[(df_special_traj['big_interval']) & (df_special_traj['valid']), 'last_timestamp']

    rest_grp = (df_special_traj.groupby('grp')['stop'].first()
                & (
                        (df_special_traj.groupby('grp')['timestamp'].last() -
                         df_special_traj.groupby('grp')['begin_time'].first()
                         ) > pd.Timedelta(minutes=30)
                   )
                )
    df_special_traj.loc[df_special_traj['grp'].isin(rest_grp.loc[rest_grp==True].index), 'rest'] = True
    df_special_traj['rest'] = df_special_traj['rest'].fillna(value=False)
    df_special_traj['new_seg'] = 0
    df_special_traj.loc[((df_special_traj['rest'] != df_special_traj['rest'].shift())
                         | (df_special_traj['plate'] != df_special_traj['plate'].shift())), 'new_seg'] = 1
    df_special_traj['grp'] = df_special_traj['new_seg'].cumsum()

    def save_non_rest(seg, path):
        if not seg.at[seg.index[0], 'rest']:
            seg[['plate', 'color', 'longitude', 'latitude', 'timestamp', 'velocity', 'dis_f_pre']]\
                .to_csv(os.path.join(path,
                                     seg.at[seg.index[0], 'plate'] + '_' + str(seg.name) + '.csv'),
                        index=False)

    seg_path = 'db/segment'
    Path(seg_path).mkdir(parents=True, exist_ok=True)
    print(trajectory.loc[(trajectory['plate'] == '粤B4BX08')
                         & (trajectory['timestamp'] > datetime.datetime(2014, 7, 16, 11, 40))])
    df_special_traj.groupby('grp').progress_apply(save_non_rest, path=seg_path,)
    ###################################################################################################################
