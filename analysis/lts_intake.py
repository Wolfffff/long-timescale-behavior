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
px_mm = 28.25  # mm/px

path_name = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/tracks"
exp1_cam1_h5s = [
    "exp2_cam1_0through23.tracked.analysis.h5",
    "exp2_cam1_24through47.tracked.analysis.h5",
    # "exp2_cam1_48through71.tracked.analysis.h5",
    # "exp2_cam1_72through95.tracked.analysis.h5",
    # "exp2_cam1_96through119.tracked.analysis.h5"
]
exp1_cam1_h5s = [path_name + "/" + filename for filename in exp1_cam1_h5s]

exp1_cam2_h5s = [
    "exp2_cam2_0through23.tracked.analysis.h5",
    "exp2_cam2_24through47.tracked.analysis.h5",
    # "exp2_cam2_48through71.tracked.analysis.h5",
    # "exp2_cam2_72through95.tracked.analysis.h5",
    # "exp2_cam2_96through119.tracked.analysis.h5"
]
exp1_cam2_h5s = [path_name + "/" + filename for filename in exp1_cam2_h5s]

expmt_dict = {
    "exp1_cam1": {
        "h5s": exp1_cam1_h5s,
        "video": "exp1_cam1.mkv",
        "frame_rate": 100,
        "start_time": datetime.strptime("0-22:33:00", FMT),
        "camera": "1",
        "experiment": "1",
        "video_path": "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/exp1/exp5_202109014_2233/Camera1/exp.mkv"
     },
        "exp1_cam2": {
        "h5s": exp1_cam2_h5s,
        "frame_rate": 100,
        "start_time": datetime.strptime("0-22:33:00", FMT),
        "camera": "2",
        "experiment": "1",
        "video_path": "test"
    }
}

tracks_dict = {}
velocities_dict = {}

for expmt_name in list(expmt_dict.keys())[0:1]:
    video_path = expmt_dict[expmt_name]["video_path"]
    keypoints, blobs, median_frame = trx_utils.blob_detector(video_path)
    arr_srt = np.array([kp.pt for kp in keypoints]).T[np.newaxis,np.newaxis,:,:]
    arr_srt[:,:,1,:] = -arr_srt[:,:,1,:]
    assignment_indices, locations, freq = trx_utils.hist_sort(arr_srt,ctr_idx=0,ymin=-1536, ymax=0)
    keypoints2 = [keypoints[i] for i in assignment_indices]
    if len(keypoints2) !=4:
        Exception("Wrong number of keypoints!")
    expmt_dict[expmt_name]['keypoints'] = [kp.pt for kp in keypoints2]
    expmt_dict[expmt_name]['keypoints_size'] = [kp.size for kp in keypoints2]

# %%
for key in list(expmt_dict.keys()):
    expmt = expmt_dict[key]
    logger.info(key)
    base_idx = 0
    with h5py.File(expmt["h5s"][0], "r") as f:
        dset_names = list(f.keys())
        node_names = [n.decode() for n in f["node_names"][:]]
        locations = f["tracks"][:].T

        locations[:, :, 1, :] = -locations[:, :, 1, :]
        assignment_indices, locations, freq = trx_utils.hist_sort(
            locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
        )
        locations[:, :, 1, :] = -locations[:, :, 1, :]

    for filename in tqdm(expmt["h5s"][1:]):
        with h5py.File(filename, "r") as f:
            temp_locations = f["tracks"][:].T
            temp_locations[:, :, 1, :] = -temp_locations[:, :, 1, :]
            temp_assignment_indices, temp_locations, freq = trx_utils.hist_sort(
                temp_locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
            )
            temp_locations[:, :, 1, :] = -temp_locations[:, :, 1, :]
            logger.info(filename)
            logger.info(freq)

            locations = np.concatenate((locations, temp_locations), axis=0)
    # Final assignment as a safety
    locations[:, :, 1, :] = -locations[:, :, 1, :]
    assignment_indices, locations, freq = trx_utils.hist_sort(
        locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
    )
    locations[:, :, 1, :] = -locations[:, :, 1, :]
    # expmt_dict[key]["tracks"] = locations
    tracks_dict[key] = locations
    expmt_dict[key]["assignments"] = assignment_indices
    expmt_dict[key]["freq"] = freq

# %%
for key in expmt_dict:
    expmt = expmt_dict[key]
    fly_node_locations_all = tracks_dict[key]
    fly_idx = 0
    indices = np.array([node_names.index("thorax")])
    fly_node_locations = fly_node_locations_all[:, :, :, [fly_idx]]
    fly_node_locations = trx_utils.fill_missing_np(fly_node_locations)
    # fly_node_locations = trx_utils.smooth_median(fly_node_locations, window=5)
    # fly_node_locations = trx_utils.smooth_gaussian(fly_node_locations, window=5)
    fly_node_velocities = trx_utils.instance_node_velocities(
        fly_node_locations, 0, fly_node_locations.shape[0]
    ) * (1/px_mm) * expmt["frame_rate"]

    for fly_idx in tqdm(range(1, fly_node_locations_all.shape[3])):
        current_fly_node_locations = fly_node_locations_all[:, :, :, [fly_idx]]
        current_fly_node_locations = trx_utils.fill_missing_np(current_fly_node_locations)
        # current_fly_node_locations = trx_utils.smooth_median(current_fly_node_locations, window=5)
        # current_fly_node_locations = trx_utils.smooth_gaussian(current_fly_node_locations, window=5)
        current_fly_node_velocities = trx_utils.instance_node_velocities(
            current_fly_node_locations, 0, current_fly_node_locations.shape[0]
        ) * (1/px_mm) * expmt["frame_rate"]
        fly_node_velocities = np.dstack((fly_node_velocities, current_fly_node_velocities))
        fly_node_locations = np.concatenate((fly_node_locations, current_fly_node_locations), axis=3)

    velocities_dict[key] = fly_node_velocities

# %%
for key in expmt_dict:
    expmt_dict[key]["node_names"] = node_names
    expmt_dict[key]["px_mm"] = px_mm
json.dump(expmt_dict, open('expmt_dict.json', 'w'),default=str)

for key in tqdm(expmt_dict):
    data_file = h5py.File(data_dir + f"/{key}_fly_node_locations.h5", 'w')
    data_file.create_dataset('tracks', data=tracks_dict[key], compression='lzf')#'gzip', compression_opts=9)
    data_file.close()

    data_file = h5py.File(data_dir + f"/{key}_fly_node_velocities.h5", 'w')
    data_file.create_dataset('velocities', data=velocities_dict[key], compression='lzf')#'gzip', compression_opts=9)
    data_file.close()

# %%
