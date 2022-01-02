# %% [markdown]
# # Caffeine base analysis

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
    # "exp2_cam1_120through143.tracked.analysis.h5",
    # "exp2_cam1_144through167.tracked.analysis.h5",
]
exp1_cam1_h5s = [path_name + "/" + filename for filename in exp1_cam1_h5s]

exp1_cam2_h5s = [
    "exp2_cam2_0through23.tracked.analysis.h5",
    "exp2_cam2_24through47.tracked.analysis.h5",
    "exp2_cam2_48through71.tracked.analysis.h5",
    "exp2_cam2_72through95.tracked.analysis.h5",
    "exp2_cam2_96through119.tracked.analysis.h5"
    # "exp2_cam2_120through143.tracked.analysis.h5",
    # "exp2_cam2_144through167.tracked.analysis.h5",
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
import json
with open('data.json', 'w') as f:
    json.dump(expmt_dict, f,default=str)

# with open('data.json', 'r') as f:
#     expmt_dict = json.load(f)

# %%
# np.save(data_dir +"/fly_node_locations_mediansmoothed.npy",fly_node_locations)
# np.save(data_dir +"/fly_node_velocities.npy",fly_node_velocities)
for key in expmt_dict:
    data_file = h5py.File(data_dir + f"/{key}_fly_node_locations.h5", 'w')
    data_file.create_dataset('tracks', data=tracks_dict[key], compression='lzf')#'gzip', compression_opts=9)
    data_file.close()

    data_file = h5py.File(data_dir + f"/{key}_fly_node_locations.h5", 'w')
    data_file.create_dataset('velocities', data=velocities_dict[key], compression='lzf')#'gzip', compression_opts=9)
    data_file.close()

tracks_dict = {}
velocities_dict = {}
for key in expmt_dict:
    with h5py.File(data_dir + f"/{key}_fly_node_locations.h5", "r") as f:
        tracks_dict[key] = f["tracks"][:]
    with h5py.File(data_dir + f"/{key}_fly_node_velocities.h5", "r") as f:
        velocities_dict[key] = f["tracks"][:]


# %%
fly_node_locations = np.concatenate([tracks_dict[key] for key in expmt_dict], axis=3)
fly_node_velocities = np.concatenate([velocities_dict[key] for key in expmt_dict], axis=2)
# fly_node_locations[:, :, 1, :] = -fly_node_locations[:, :, 1, :]
# assignment_indices, locations, freq = trx_utils.hist_sort(
#     fly_node_locations, ctr_idx=node_names.index("thorax"), ymin=-1536, ymax=0
# )
# fly_node_locations[:, :, 1, :] = -fly_node_locations[:, :, 1, :]

# %%
import scipy.stats
from datetime import datetime
frame_rate = expmt_dict[list(expmt_dict.keys())[0]]['frame_rate']
FMT = '%w-%H:%M:%S'

time = datetime.strptime('0-22:33:00', FMT)
start_day = datetime.strptime('1-00:00:00', FMT) # for example
difference = start_day - time 
shift = int(difference.seconds*frame_rate)

plt.rcParams["figure.figsize"] = (9,3)
start_frame = 0
end_frame = int(24*60*60*frame_rate)
data = fly_node_velocities[:,node_names.index("thorax"),:].copy()
frame_idx = (np.arange(data.shape[0]) - shift)
ToD = frame_idx % int(24*60*60*frame_rate) / (1*60*60*frame_rate)

# ToD = ToD[int(6*60*60*frame_rate):]
# data = data[int(6*60*60*frame_rate):,:]

plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'
# fig, ax = plt.subplots(figsize=(9, 3), dpi=300)
data[data < 0.1] = 0
for fly_idx in range(data.shape[1]):
    binned = scipy.stats.binned_statistic(ToD, data[:,fly_idx],statistic='sum', bins=72, range=None)

    plt.bar(x=np.arange(72),height=binned.statistic,alpha=0.2,width=1)
    plt.xticks(rotation=60)
    # plt.axvspan(xmin=7.5,xmax=19.5,alpha=0.2)
    plt.tight_layout()
    plt.show()
# %%
import scipy.stats
from datetime import datetime
frame_rate = expmt_dict[list(expmt_dict.keys())[0]]['frame_rate']
FMT = '%w-%H:%M:%S'

time = datetime.strptime('0-22:33:00', FMT)
start_day = datetime.strptime('1-00:00:00', FMT) # for example
difference = start_day - time 
shift = int(difference.seconds*frame_rate)

plt.rcParams["figure.figsize"] = (9,3)
start_frame = 0
end_frame = int(24*60*60*frame_rate)
data = fly_node_velocities[:,node_names.index("thorax"),:].copy()
frame_idx = (np.arange(data.shape[0]) - shift)
ToD = frame_idx % int(24*60*60*frame_rate) / (1*60*60*frame_rate)

data[data < 0.5] = 0
for fly_idx in range(data.shape[1]):
    data[:,fly_idx] = data[:,fly_idx] / np.nansum(data[:,fly_idx])

binned = scipy.stats.binned_statistic(np.repeat(ToD,data.shape[1]), data.flatten(),statistic='mean',
 bins=72, range=None)

plt.bar(x=np.arange(72),height=binned.statistic,alpha=1,width=1)
# plt.yscale('log')
plt.xticks(rotation=60)
plt.axvspan(xmin=7.5,xmax=19.5,alpha=0.2)
plt.tight_layout()
plt.show()
# %%
data[data < 0.1] = 0
for fly_idx in range(data.shape[1]):
    binned = scipy.stats.binned_statistic(ToD, data[:,fly_idx],statistic='sum', bins=72, range=None)

    plt.bar(x=np.arange(72),height=binned.statistic,alpha=0.2,width=1)
    plt.xticks(rotation=60)
    # plt.axvspan(xmin=7.5,xmax=19.5,alpha=0.2)
    plt.tight_layout()
    plt.show()

# %%
from cv2 import cv2
thorax_loc = fly_node_locations[0:(48*60*60*100),node_names.index("thorax"),:,:]
cap = cv2.VideoCapture('/Genomics/ayroleslab2/scott/long-timescale-behavior/data/exp1/exp5_202109014_2233/Camera1/exp.mkv')
# for i in range(4):
i=0
data = thorax_loc[:,:,i]
x = data[:,0]
y = data[:,1]
plt.hist2d(x, y, norm=mpl.colors.LogNorm())
plt.axis('equal')
plt.title("Simple 2D Histogram")
plt.show()
mid_pt = ((np.max(x) + np.min(x))/2, (np.max(y) + np.min(y))/2)
relative_pos = data-mid_pt
dist = np.linalg.norm(relative_pos,axis=1)
# plt.hist(dist)
plt.plot(x[dist < 310], y[dist < 310],alpha=0.2)
cap.set(cv2.CAP_PROP_POS_FRAMES,np.where(dist>320)[0][0])
res, frame = cap.read()
frame = frame[:, :, 0]
plt.imshow(frame)
plt.show()
logger.info(f'FRAC ON EDGE: {x[dist > 320].shape[0]/x.shape[0] * 100}%')

# %%


from matplotlib.colors import LogNorm, Normalize
subset = np.random.choice(np.arange(x.shape[0]),100000,replace=False)
plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
# plt.hist2d(x[subset],y[subset], bins=24,norm=LogNorm(), cmap='plasma')
hmax = sns.kdeplot(x[subset],y[subset],cmap = trx_utils.alpha_cmap(plt.cm.viridis),
   shade = True,
   shade_lowest = False,
   antialiased=True,
   n_levels = 20,
   cbar=True)
# 
# #cbar=True,shade=True, cmap=alpha_cmap(plt.cm.plasma),levels=20,bw_adjust=.5,norm = LogNorm(),antialiased=True)
# # hmax.collections[0].set_alpha(0)
plt.xlim(0,1536/2)
plt.ylim(1536/2,1536)
plt.show()


# %%

from cv2 import cv2
thorax_loc = fly_node_locations[0:(48*60*60*100),node_names.index("thorax"),:,:]
cap = cv2.VideoCapture('/Genomics/ayroleslab2/scott/long-timescale-behavior/data/exp1/exp5_202109014_2233/Camera1/exp.mkv')

i=3
data = thorax_loc[:,:,i]
x = data[:,0]
y = data[:,1]
plt.hist2d(x, y, norm=mpl.colors.LogNorm())
plt.axis('equal')
plt.title("Simple 2D Histogram")
plt.show()
mid_pt = ((np.max(x) + np.min(x))/2, (np.max(y) + np.min(y))/2)
relative_pos = data-mid_pt
dist = np.linalg.norm(relative_pos,axis=1)

data_wall = thorax_loc[dist > 320,:,i]
cap.set(cv2.CAP_PROP_POS_FRAMES,np.where(dist>320)[0][0])
x_wall = data_wall[:,0]
y_wall = data_wall[:,1]
subset = np.random.choice(np.arange(data_wall.shape[0]),1000,replace=False)
# plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
# plt.hist2d(x[subset],y[subset], bins=24,norm=LogNorm(), cmap='plasma')
# hmax = sns.kdeplot(x_wall[subset],y_wall[subset],cmap = trx_utils.alpha_cmap(plt.cm.viridis),
#    shade = True,
#    shade_lowest = False,
#    antialiased=True,
#    n_levels = 20,
#    cbar=True)
   
# 
# #cbar=True,shade=True, cmap=alpha_cmap(plt.cm.plasma),levels=20,bw_adjust=.5,norm = LogNorm(),antialiased=True)
# # hmax.collections[0].set_alpha(0)
# plt.xlim(0,1536/2)
# plt.ylim(1536/2,1536)
# plt.show()
# %%
n = 0
for i in np.argsort(-dist)[::10000]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    res, frame = cap.read()
    frame = frame[:, :, 0]
    print(n)
    print(i)
    print(dist[i])
    n += 10000
    plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
    plt.show()

# %%
cap = cv2.VideoCapture('/Genomics/ayroleslab2/scott/long-timescale-behavior/data/exp1/exp5_202109014_2233/Camera1/exp.mkv')
res, frame = cap.read()
im = frame[:, :, 0]
plt.imshow(im)
plt.show()
th, im_th = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)
# Copy the thresholded image.

im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_out = im_th | im_floodfill_inv

plt.hist(im_th.flatten())
plt.show()
plt.imshow(im_out, cmap='gray', vmin=0, vmax=255)
plt.show()

image = im_out

params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = False
params.minArea = 250000
 
# Set Circularity filtering parameters
params.filterByCircularity = False
params.minCircularity = 0.8
 
# Set Convexity filtering parameters
params.filterByConvexity = False
params.minConvexity = 0.2
     
# Set inertia filtering parameters
params.filterByInertia = False
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
     
# Detect blobs
keypoints = detector.detect(255-image)
 
# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("./blobs.png", blobs)
plt.imshow(blobs)#,# cmap='gray', vmin=0, vmax=255)
# Show blobs
# %%
dir(keypoints[0])
keypoints[0]

# %%
import utils.trx_utils as trx_utils

# %%
# import utils.trx_utils as trx_utils
import importlib
import utils.trx_utils
import cv2
importlib.reload(utils.trx_utils)
a,b,c = utils.trx_utils.blob_detector("/Genomics/ayroleslab2/scott/long-timescale-behavior/data/exp1/exp5_202109014_2233/Camera1/exp.mkv")
cv2.imwrite("./blobs.png", b)
cv2.imwrite("./blobs_c.png", c)
# %%
