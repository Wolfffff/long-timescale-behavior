# %%
# %% [markdown]
# # lts analysis

# %%
import logging
from seaborn.distributions import distplot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import palettable
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import pandas as pd
import joypy
import h5py
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime
import importlib

FMT = "%w-%H:%M:%S"

wd = "/Genomics/ayroleslab2/scott/long-timescale-behavior/analysis"
os.chdir(wd)

import utils.trx_utils as trx_utils

data_dir = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data"

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis_logger")

# %%
expmt_dict = json.load(open('expmt_dict.json','r'))
node_names =expmt_dict["exp1_cam1"]['node_names']
px_mm = expmt_dict["exp1_cam1"]['px_mm']
tracks_dict = {}
velocities_dict = {}
for key in tqdm(expmt_dict):
    with h5py.File(data_dir + f"/{key}_fly_node_locations.h5", "r") as f:
        tracks_dict[key] = f["tracks"][:]
    with h5py.File(data_dir + f"/{key}_fly_node_velocities.h5", "r") as f:
        velocities_dict[key] = f["velocities"][:]
        
# %%
import scipy.stats
from datetime import datetime, timedelta
ToD = {}
for expmt in expmt_dict:
    expmt_data = expmt_dict[expmt]
    FMT = '%w-%H:%M:%S'
    start_day = datetime.strptime('1900-01-01 00:00:00', '%Y-%m-%d %H:%M:%S') # for example
    try:
        time = datetime.strptime(expmt_data['start_time'],'%Y-%m-%d %H:%M:%S')
    except:
        time = expmt_data['start_time']
    expmt_data['start_time'] = time
    difference = start_day - time
    frame_rate = expmt_data['frame_rate']
    shift = int(difference.seconds*frame_rate)
    frame_idx = (np.arange(tracks_dict[expmt].shape[0]) - shift)
    expmt_tod = (frame_idx % int(24*60*60*frame_rate)) / (1*60*60*frame_rate)
    ToD[expmt] = expmt_tod

day_start = 8
day_end = 20
day_dict = {}
for expmt in expmt_dict:
    day_dict[expmt] = (ToD[expmt] > day_start) & (ToD[expmt] < day_end)

plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams["figure.figsize"] = (9,3)
plt.rcParams['figure.dpi'] = 300

# %%

for expmt in expmt_dict:
    vels = velocities_dict[expmt].copy()
    day = (ToD[expmt] > day_start) & (ToD[expmt] < day_end)
    for fly_idx in range(vels.shape[2]):
        fly_thorax_vel = vels[:,node_names.index("thorax"),fly_idx]
        # fly_thorax_vel[fly_thorax_vel < 3*(1/28.25)*100] = 0
        binned = scipy.stats.binned_statistic(day.astype(int), fly_thorax_vel,statistic='mean', bins=[0,.5,1], range=None)
        sns.barplot(x=["Night","Day"], y=binned.statistic)
        plt.title(f'{expmt} - Fly {fly_idx} - Mean thorax velocity by day/night')
        plt.show()
        
        sns.boxplot(x=day.astype(int),y=fly_thorax_vel,showfliers=False)
        plt.title(f'{expmt} - Fly {fly_idx} - Mean thorax velocity by day/night')
        plt.show()
        logger.info(f"{expmt} {fly_idx}")
        segments = 1*24

        binned = scipy.stats.binned_statistic(ToD[expmt], fly_thorax_vel,statistic='mean', bins=segments, range=None)
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        fig, ax = plt.subplots()
        sns.barplot(ax=ax,x=np.arange(segments),y=binned.statistic,color=palettable.wesanderson.GrandBudapest4_5.mpl_colors[0])#,alpha=1,width=1)
        plt.xticks([i*(segments//24) for i in [0, 8, 20,24]], [0, 8, 20,24])
        plt.tight_layout(pad=2)
        # ax.set_yscale('log')
        def change_width(ax, new_value) :
            for patch in ax.patches :
                current_width = patch.get_width()
                diff = current_width - new_value

                # we change the bar width
                patch.set_width(new_value)

                # we recenter the bar
                patch.set_x(patch.get_x() + diff * .5)
        plt.xlabel("Hour of the day")
        plt.ylabel("Mean thorax velocity (mm/s)")
        plt.title("Mean thorax velocity by hour of the day")
        change_width(ax, .95)
        plt.savefig(f'{expmt}_fly{fly_idx}_thorax_velocity_by_hour_of_day.png')
        plt.show()

# %%
# 
day_start = 8
day_end = 20
day_map = {0: "Night", 1: "Day"}
for expmt in expmt_dict:
    logger.info(f'{expmt}')
    vels = velocities_dict[expmt].copy()
    # data[data < .1] = 0
    ToD_list = []
    day_list = []
    fly_thorax_vels_list = []
    for fly_idx in range(4):
        logger.info(f'{expmt} {fly_idx}')
        fly_thorax_vels_list.append(vels[:,node_names.index("thorax"),fly_idx])
        ToD_list.append(ToD[expmt])
        day_list.append(day_dict[expmt])
        
    ToD_mat = np.concatenate(ToD_list)
    day_mat = np.concatenate(day_list)
    vel_mat = np.concatenate(fly_thorax_vels_list)
    binned = scipy.stats.binned_statistic(day_mat.astype(int), vel_mat,statistic='mean', bins=[0,.5,1])

    sns.barplot(x=np.arange(2),y=binned.statistic)#,alpha=1,width=1)
    plt.title(f'{expmt} - Mean thorax velocity by day/night')
    # plt.title(f'{expmt} - Frac time moving by day/night')
    plt.show()
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    plt.plot(ToD_mat,vel_mat)
    # df = pd.DataFrame({'x': ToD_mat, 'y': vel_mat})
    # sns.boxplot(x='x', y='y', data=df,showfliers=True)
    plt.title(f'{expmt} - Moving/not moving by day/night')
    # plt.xlim(0,10)
    plt.show()
# %%
import cv2
expmt = 'exp1_cam1'
expmt_data = expmt_dict[expmt]
cap=cv2.VideoCapture(expmt_data['video_path'])
fast_idx = np.where(velocities_dict[expmt][:,node_names.index("thorax"),0] > 20*(1/28.25)*100)[0]
for idx in fast_idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES,idx-4)
    logger.info(f'{idx}')
    logger.info(f'{velocities_dict[expmt][idx,node_names.index("thorax"),0]}')
    for i in range(3):
        res, frame = cap.read()
        frame = frame[:, :, 0]
        fig, ax = plt.subplots()
        a = plt.imshow(frame, cmap='gray', vmin=0, vmax=255);
        b = plt.scatter(x=tracks_dict[expmt][int(cap.get(cv2.CAP_PROP_POS_FRAMES)),node_names.index("thorax"),0,0],y=tracks_dict[expmt][int(cap.get(cv2.CAP_PROP_POS_FRAMES)),node_names.index("thorax"),1,0],s=10,c='r');
        plt.savefig(f'ugh_{expmt}_frame{cap.get(cv2.CAP_PROP_POS_FRAMES)}_thorax_velocity.png')

# %%
