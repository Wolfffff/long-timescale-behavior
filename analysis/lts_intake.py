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


px_mm = 28.25  # mm/px

path_name = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/tracks"
exp1_cam1_h5s = [
    "exp2_cam1_0through23.tracked.analysis.h5",
    "exp2_cam1_24through47.tracked.analysis.h5",
    "exp2_cam1_48through71.tracked.analysis.h5",
    "exp2_cam1_72through95.tracked.analysis.h5",
    "exp2_cam1_96through119.tracked.analysis.h5"
]
exp1_cam1_h5s = [path_name + "/" + filename for filename in exp1_cam1_h5s]

exp1_cam2_h5s = [
    "exp2_cam2_0through23.tracked.analysis.h5",
    "exp2_cam2_24through47.tracked.analysis.h5",
    "exp2_cam2_48through71.tracked.analysis.h5",
    "exp2_cam2_72through95.tracked.analysis.h5",
    "exp2_cam2_96through119.tracked.analysis.h5"
]
exp1_cam2_h5s = [path_name + "/" + filename for filename in exp1_cam2_h5s]

expmt_dict = {
    "exp1_cam1": {
        "h5s": exp1_cam1_h5s,
        "video": "exp1_cam1.mkv",
        "frame_rate": 100,
        "start_time": datetime.strptime("0-22:33:00", FMT),
        "camera": "1",
        "experiment": "1"
    },
        "exp1_cam2": {
        "h5s": exp1_cam2_h5s,
        "frame_rate": 100,
        "start_time": datetime.strptime("0-22:33:00", FMT),
        "camera": "2",
        "experiment": "1"
    }
}

tracks_dict = {}
velocities_dict = {}
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
            print(filename)
            print(freq)

            locations = np.concatenate((locations, temp_locations), axis=0)
    # Final assignment as a safety
    locations[:, :, 1, :] = -locations[:, :, 1, :]
    assignment_indices, locations, freq = trx_utils.hist_sort(
        locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
    )
    locations[:, :, 1, :] = -locations[:, :, 1, :]
    locations = trx_utils.fill_missing_np(locations)
    # expmt_dict[key]["tracks"] = locations
    tracks_dict[key] = locations
    expmt_dict[key]["assignments"] = assignment_indices
    expmt_dict[key]["freq"] = freq

HEAD_INDEX = node_names.index("head")
THORAX_INDEX = node_names.index("thorax")
ABDO_INDEX = node_names.index("abdomen")
# %%
for key in expmt_dict:
    expmt = expmt_dict[key]
    fly_node_locations_all = tracks_dict[key]
    fly_idx = 0
    indices = np.array([THORAX_INDEX])
    fly_node_locations = fly_node_locations_all[:, :, :, [fly_idx]]
    fly_node_locations = trx_utils.smooth_median(fly_node_locations, window=5)
    fly_node_velocities = trx_utils.instance_node_velocities(
        fly_node_locations, 0, fly_node_locations.shape[0]
    ).astype(np.float32) * (1/px_mm) * expmt["frame_rate"]

    for fly_idx in tqdm(range(1, fly_node_locations_all.shape[3])):
        current_fly_node_locations = fly_node_locations_all[:, :, :, [fly_idx]]
        current_fly_node_locations = trx_utils.smooth_median(current_fly_node_locations, window=5)
        current_fly_node_velocities = trx_utils.instance_node_velocities(
            current_fly_node_locations, 0, current_fly_node_locations.shape[0]
        ).astype(np.float32) * (1/px_mm) * expmt["frame_rate"]
        fly_node_velocities = np.dstack((fly_node_velocities, current_fly_node_velocities))
        fly_node_locations = np.concatenate((fly_node_locations, current_fly_node_locations), axis=3)

    velocities_dict[key] = fly_node_velocities

# %%
with open('data.json', 'w') as f:
    json.dump(expmt_dict, f,default=str)

# %%
for key in tqdm(expmt_dict):
    data_file = h5py.File(data_dir + f"/{key}_fly_node_locations.h5", 'w')
    data_file.create_dataset('tracks', data=tracks_dict[key], compression='lzf')#'gzip', compression_opts=9)
    data_file.close()

    data_file = h5py.File(data_dir + f"/{key}_fly_node_velocities.h5", 'w')
    data_file.create_dataset('velocities', data=velocities_dict[key], compression='lzf')#'gzip', compression_opts=9)
    data_file.close()

