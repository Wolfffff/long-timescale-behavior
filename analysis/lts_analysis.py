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
ToD = {}
for expmt in expmt_dict:
    expmt_data = expmt_dict[expmt]
    FMT = '%w-%H:%M:%S'
    start_day = datetime.strptime('1900-01-02 00:00:00', '%Y-%m-%d %H:%M:%S') # for example
    time = expmt_data['start_time']#,'%Y-%m-%d %H:%M:%S')
    expmt_data['start_time'] = time
    difference = start_day - time 
    frame_rate = expmt_data['frame_rate']
    shift = int(difference.seconds*frame_rate)
    frame_idx = (np.arange(tracks_dict[expmt].shape[0]) - shift)
    expmt_tod = frame_idx % int(24*60*60*frame_rate) / (1*60*60*frame_rate)
    ToD[expmt] = expmt_tod


plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'
# fig, ax = plt.subplots(figsize=(9, 3), dpi=300)
# data[data < 0.1] = 0

# %%
for expmt in expmt_dict:
    # data[data < .1] = 0

    vels = velocities_dict[expmt]
    for fly_idx in range(vels.shape[2]):
        vels[:,node_names.index("thorax"),:] = vels[:,node_names.index("thorax"),:] / np.nansum(vels[:,node_names.index("thorax"),:])
    for fly_idx in range(vels.shape[2]):
        segments = 2*24
        binned = scipy.stats.binned_statistic(ToD[expmt], vels[:,node_names.index("thorax"),fly_idx],statistic='mean', bins=segments, range=None)
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        fig, ax = plt.subplots()
        sns.barplot(ax=ax,x=np.arange(segments),y=binned.statistic,color=palettable.wesanderson.GrandBudapest4_5.mpl_colors[0])#,alpha=1,width=1)
        plt.xticks([i*(segments//24) for i in [0, 8, 20,24]], [0, 8, 20,24])
        plt.tight_layout()
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
        plt.show()
# %%
# flattened
for expmt in expmt_dict:
    logger.info(expmt)
    vels = velocities_dict[expmt]
    ToD_mat = np.concatenate([np.repeat(ToD[expmt][:,np.newaxis],4,axis=1)], axis=1)
    for fly_idx in range(vels.shape[2]):
            vels[:,node_names.index("thorax"),:] = vels[:,node_names.index("thorax"),:] / np.nansum(vels[:,node_names.index("thorax"),:])

    segments = 2*24
    binned = scipy.stats.binned_statistic(ToD_mat.flatten(), vels[:,node_names.index("thorax"),:].flatten(),statistic='mean', bins=segments, range=None)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    fig, ax = plt.subplots()
    sns.barplot(ax=ax,x=np.arange(segments),y=binned.statistic,color=palettable.wesanderson.GrandBudapest4_5.mpl_colors[0])#,alpha=1,width=1)
    plt.xticks([i*(segments//24) for i in [0, 8, 20,24]], [0, 8, 20,24])
    plt.tight_layout()
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
data = fly_node_velocities[:,node_names.index("thorax"),0:2].copy()
frame_idx = (np.arange(data.shape[0]) - shift)
ToD = frame_idx % int(24*60*60*frame_rate) / (1*60*60*frame_rate)
plt.rcParams["figure.figsize"] = (9,3)
plt.rcParams['figure.dpi'] = 300
# data[data < .1] = 0
for fly_idx in range(data.shape[1]):
    data[:,fly_idx] = data[:,fly_idx] / np.nansum(data[:,fly_idx])

binned = scipy.stats.binned_statistic(np.repeat(ToD,data.shape[1]), data.flatten(),statistic='mean',
 bins=24*3, range=None)



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
ToD = frame_idx / (1*60*60*frame_rate)
plt.rcParams["figure.figsize"] = (9,3)
plt.rcParams['figure.dpi'] = 300

binned = scipy.stats.binned_statistic(np.repeat(ToD,data.shape[1]), data.flatten(),statistic='mean',
 bins=10000*5*24, range=None)
# %%
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
# data = fly_node_velocities[1:10000000,node_names.index("thorax"),0]

f, t, Sxx = signal.spectrogram(binned.statistic,fs=1000,window=('tukey', 10))#,nperseg=4,noverlap=3,nfft=128,detrend='constant',return_onesided=True, scaling='density')
plt.pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
# plt.xlim(0,100)
plt.show()
# %%
x = binned.statistic
Xf_mag = np.abs(np.fft.fft(x))
freqs = np.fft.fftfreq(len(Xf_mag), d=1.0/24)
import matplotlib.pyplot as plt
plt.plot(freqs, Xf_mag)
plt.show()
# %%
# data[data < 0.1] = 0
for fly_idx in range(data.shape[1]):
    data = fly_node_velocities[:,node_names.index("thorax"),:].copy()
    frame_idx = (np.arange(data.shape[0]) - shift)
    ToD = frame_idx % int(24*60*60*frame_rate) / (1*60*60*frame_rate)

    binned = scipy.stats.binned_statistic(ToD, data[:,fly_idx],statistic='sum', bins=48, range=None)
    sns.barplot(x=np.arange(48),y=binned.statistic)#,alpha=0.2,width=1)
    plt.xticks(rotation=60)
    # plt.axvspan(xmin=7.5,xmax=19.5,alpha=0.2)
    plt.tight_layout()
    plt.show()

# %%
for i in expmt_dict:
    data = thorax_loc[:,:,i]
    x = data[:,0]
    y = data[:,1]
    plt.hist2d(x, y, norm=mpl.colors.LogNorm())
    plt.axis('equal')
    plt.title("Simple 2D Histogram")
    plt.show()
    # mid_pt = ((np.max(x) + np.min(x))/2, (np.max(y) + np.min(y))/2)
    mid_pt = np.array(keypoints2[i].pt)
    relative_pos = data-mid_pt
    dist = np.linalg.norm(relative_pos,axis=1)
    thresh = (keypoints[i].size/2) - (px_mm*1.5)
    logger.info(f'Threshold: {thresh}')


    data_wall = thorax_loc[dist > thresh,:,i]
    cap.set(cv2.CAP_PROP_POS_FRAMES,np.where(dist>thresh)[0][0])
    x_wall = data_wall[:,0]
    y_wall = data_wall[:,1]
    subset = np.random.choice(np.arange(data_wall.shape[0]),1000,replace=False)
    logger.info(f'FRAC ON EDGE: {np.where(dist > thresh)[0].shape[0]/dist.shape[0] * 100}%')
# %%

def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=3, length_includes_head=True)

n = 0
step = 10
dist_threshed = dist
dist_threshed[dist_threshed > thresh] = np.nan
cap=cv2.VideoCapture(video_path)
for i in (dist_threshed).argsort()[:100:step]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i-1)
    res, frame = cap.read()
    frame = frame[:, :, 0]
    print(n)
    print(i)
    print(dist_threshed[i])
    n += step
    fig, ax = plt.subplots()
    plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
    for keypoint in keypoints2:
        ax.add_artist( plt.Circle(keypoint.pt, thresh, color='r', fill=False) )
    drawArrow(keypoints2[0].pt,data[i,:])
        # ax.add_artist( plt.Circle(keypoint.pt, dist[i], color='b', fill=False) )
    
    plt.show()

    

# %%


fig, ax = plt.subplots()
ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
for keypoint in keypoints2:
    ax.add_artist( plt.Circle(keypoint.pt, thresh, color='r', fill=False) )
plt.show()
logger.info(keypoints)
# %%
