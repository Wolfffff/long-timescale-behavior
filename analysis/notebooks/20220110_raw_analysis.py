# %% [markdown]
# # Raw track analysis notebook
# A notebook to analyze the raw trace data and find a reasonable set of smoothing parameters.

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

wd = "/Genomics/ayroleslab2/scott/long-timescale-behavior/analysis/"
os.chdir(wd)

import utils.trx_utils as trx_utils

data_dir = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/"
track_dir = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/tracks/"

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis_logger")


# %% [markdown]
# # Load the h5 traces and sort by quadrant
# This uses hist_sort to assign each of the traces. Because the coordinate system is different, the y-axis is flipped to be in the same coordinate system as the image when sorted. That is, flip the y-xis to get (0,0) in the top left corner of our space.

# %%
exp1_cam1_h5s = [
    "exp2_cam1_0through23.tracked.analysis.h5",
    # "exp2_cam1_24through47.tracked.analysis.h5",
    # "exp2_cam1_48through71.tracked.analysis.h5",
    # "exp2_cam1_72through95.tracked.analysis.h5",
    # "exp2_cam1_96through119.tracked.analysis.h5",
]
exp1_cam1_h5s = [track_dir + filename for filename in exp1_cam1_h5s]

bright_h5s = [
    "24h_bright_0through23_0through29.tracked.analysis.h5"
]
bright_dir = "/Genomics/ayroleslab2/scott/long-timescale-behavior/tmp/24h_bright/"
bright_h5s = [track_dir + filename for filename in bright_h5s]


FMT = "%w-%H:%M:%S"

# Build with dict for compatibility with JSON
expmt_dict = {
    "exp1_cam1": {
        "h5s": exp1_cam1_h5s,
        "video": "exp1_cam1.mkv",
        "frame_rate": 100,
        "start_time": datetime.strptime("0-22:33:00", FMT),
        "camera": "1",
        "experiment": "1",
        "video_path": "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/exp1/exp5_202109014_2233/Camera1/exp.mkv",
        "px_mm": 28.25,
    },
    "exp1_cam1": {
        "h5s": bright_h5s,
        "video": "/Genomics/ayroleslab2/scott/long-timescale-behavior/tmp/24h_bright/24h_bright.mkv",
        "frame_rate": 99.96,
        "start_time": datetime.strptime("0-12:00:00", FMT),
        "camera": "1",
        "experiment": "1",
        "video_path": "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/exp1/exp5_202109014_2233/Camera1/exp.mkv",
        "px_mm": 28.25,
    },
    
}
px_mm = expmt_dict["exp1_cam1"]['px_mm']

tracks_dict = {}
velocities_dict = {}

# %% [markdown]
# # Load h5 traces and match by quadrant
# For reference, fly node locations are in the form (time, node, coord, fly_idx)

# %%
for key in expmt_dict:
    expmt_name = str(key)
    logger.info(f"Loading {expmt_name}")
    expmt = expmt_dict[key]

    with h5py.File(expmt["h5s"][0], "r") as f:
        logger.info(expmt["h5s"][0])
        dset_names = list(f.keys())
        # Note the assignment of node_names here!
        node_names = [n.decode() for n in f["node_names"][:]]
        locations = f["tracks"][:].T

        locations[:, :, 1, :] = -locations[:, :, 1, :]
        assignment_indices, locations, freq = trx_utils.hist_sort(
            locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
        )
        locations[:, :, 1, :] = -locations[:, :, 1, :]

    if len(expmt["h5s"]) > 1:
        for filename in tqdm(expmt["h5s"][1:]):
            with h5py.File(filename, "r") as f:
                temp_locations = f["tracks"][:].T
                temp_locations[:, :, 1, :] = -temp_locations[:, :, 1, :]
                temp_assignment_indices, temp_locations, freq = trx_utils.hist_sort(
                    temp_locations,
                    ctr_idx=node_names.index("thorax"),
                    ymin=-1536,
                    ymax=0,
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
    logger.info(f'Experiment: {str(expmt)}')
    logger.info(f"Final frequencies: {freq}")
    logger.info(f"Final assignments: {assignment_indices}")

    tracks_dict[expmt_name] = locations#[0:1000,:,:,:]
    expmt_dict[expmt_name]["assignments"] = assignment_indices
    expmt_dict[expmt_name]["freq"] = freq

# %%
for key in expmt_dict:
    expmt = expmt_dict[key]
    fly_node_locations = tracks_dict[key]
    fly_node_locations = trx_utils.fill_missing_np(fly_node_locations)

    fly_node_locations = trx_utils.smooth_median(fly_node_locations, window=5)
    fly_node_locations = trx_utils.smooth_gaussian(fly_node_locations, window=5)
    
    fly_node_velocities = trx_utils.instance_node_velocities(
            fly_node_locations, 0, fly_node_locations.shape[0]
        ) * (1/px_mm) * expmt["frame_rate"]
    tracks_dict[key] = fly_node_locations#_filled
    velocities_dict[key] = fly_node_velocities

# %% [markdown]
# # Save
# Save the JSON and h5s if needed.

# %%
# json.dump(expmt_dict, open('expmt_dict.json', 'w'),default=str)

# for key in tqdm(expmt_dict):
#     data_file = h5py.File(data_dir + f"{key}_fly_node_locations.h5", 'w')
#     data_file.create_dataset('tracks', data=tracks_dict[key])#, compression='lzf')#'gzip', compression_opts=9)
#     data_file.close()

#     data_file = h5py.File(data_dir + f"{key}_fly_node_velocities.h5", 'w')
#     data_file.create_dataset('velocities', data=velocities_dict[key])#, compression='lzf')#'gzip', compression_opts=9)
#     data_file.close()


# %%
importlib.reload(trx_utils)
expmt_name = 'exp1_cam1'
frame_start = int(27140*100)
frame_end = int(27140*100 + 600*100)

trx_utils.plot_trx(tracks_dict[expmt_name],expmt_dict[expmt_name]["video_path"],frame_start,frame_end,output_path="working_plot")
# %%
importlib.reload(trx_utils)
expmt_name = 'exp1_cam1'
frame_start = int(27145*100)
frame_end = int(27145*100 + 600*100)

for fly_id in range(4):
    egocentric_node_locations, egocentric_angles = trx_utils.normalize_to_egocentric(tracks_dict[expmt_name][:,:,:,fly_id], ctr_ind=node_names.index("abdomen"),fwd_ind=node_names.index("head"),return_angles=True)
    # egocentric_angles=None
    trx_utils.plot_ego(tracks_dict[expmt_name],expmt_dict[expmt_name]["video_path"],egocentric_angles,[fly_id],node_names.index("thorax"),frame_start,frame_end,output_path=f'{expmt_name}_raw_ego_{fly_id}.mp4')
    # egocentric_velocities = trx_utils.instance_node_velocities(
    #         egocentric_node_locations, 0, egocentric_node_locations.shape[0]
    #     ) * (1/px_mm) * expmt["frame_rate"]
# %%
