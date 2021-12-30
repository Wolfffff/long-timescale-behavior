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

frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px

path_name = "/Genomics/ayroleslab2/scott/long-timescale-behavior/lts/tracks"

exp1_cam1_h5s = ["exp2_cam1_0through23.tracked.analysis.h5", "exp2_cam1_24through47.tracked.analysis.h5",
"exp2_cam1_48through71.tracked.analysis.h5", "exp2_cam1_72through95.tracked.analysis.h5",
"exp2_cam1_96through119.tracked.analysis.h5","exp2_cam1_120through143.tracked.analysis.h5",
 "exp2_cam1_144through167.tracked.analysis.h5"]

 

#  exp1_cam2_h5s = []

exp1_cam1_h5s = [path_name + "/" + filename for filename in exp1_cam1_h5s]

exp1_cam1_videos = ["data/videos/0through23_cam1.mp4","data/videos/0through23_cam2.mp4"]

experiments = [exp1_cam1_h5s]

expmt_dict = {"exp1_cam1": exp1_cam1_h5s}

# data_index = pd.read_csv()
video_info = pd.DataFrame(columns = ["track_idx","assignment"])
# %%

max_frames = 0
for experiment in experiments:
    running_sum = 0
    for file in experiment:
        with h5py.File(file, "r") as f:
            running_sum += f["tracks"].shape[3]
    if running_sum > max_frames:
        max_frames = running_sum

for experiment in experiments:
    with h5py.File(experiment[0], "r") as f:
        node_names = [n.decode() for n in f["node_names"][:]]

# tracks = np.empty((max_frames,14,2,len(experiments*2)))
# tracks[:] = np.nan
# %%
base_idx = 0
for experiment in experiments:
    with h5py.File(experiment[0], "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
        locations[:,:,1,:] = -locations[:,:,1,:]
        assignment_indices, locations, freq = trx_utils.hist_sort(
            locations, ctr_idx=node_names.index("thorax"), ymin=-1536,ymax=0
        )
        locations[:,:,1,:] = -locations[:,:,1,:]

    for filename in tqdm(experiment[1:]):
        with h5py.File(filename, "r") as f:
            temp_locations = f["tracks"][:].T
            temp_locations[:,:,1,:] = -temp_locations[:,:,1,:]
            temp_assignment_indices, temp_locations, freq = trx_utils.hist_sort(
                temp_locations, ctr_idx=node_names.index("thorax"), ymin=-1536,ymax=0
            )
            temp_locations[:,:,1,:] = -temp_locations[:,:,1,:]

            locations = np.concatenate((locations, temp_locations), axis=0)
            logger.info(filename)
            logger.info(freq)
    # Final assignment as a safety
    locations[:,:,1,:] = -locations[:,:,1,:]
    assignment_indices, locations,freq  = trx_utils.hist_sort(
        locations, ctr_idx=node_names.index("thorax"),  ymin=-1536,ymax=0
    )
    locations[:,:,1,:] = -locations[:,:,1,:]
    logger.info(freq)
    # for i in range(locations.shape[3]):
    #     video_info = video_info.append({'track_idx' : base_idx, 'assignment':assignment_indices[i]}, 
    #                 ignore_index = True)
    #     base_idx = base_idx + 1
    # p = Path(experiment[0])
    # p = p.with_suffix('')
    # np.save(p, locations)
        # tracks[:,:,:,base_idx] = locations[:,:,:,base_idx]

# Drop proboscis
# locations = np.delete(locations, node_names.index("proboscis"), axis=1)
# locations = np.load("")

for fly_idx in range(locations.shape[3]):
    np.save(data_dir + f'/raw_tracks_fly_{fly_idx}.npy',locations[:,:,:,fly_idx])

locations = trx_utils.fill_missing_np(locations)


print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")

frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

HEAD_INDEX = node_names.index("head")
THORAX_INDEX = node_names.index("thorax")
ABDO_INDEX = node_names.index("abdomen")

# %%
import matplotlib.pyplot as plt
# data = locations[, :, :, fly_idx]
fly_idx = 0
indices = np.array([THORAX_INDEX])
fly_node_locations = locations[:, :, :, [fly_idx]]
fly_node_locations = trx_utils.smooth_median(fly_node_locations, window=5)
fly_node_velocities = trx_utils.instance_node_velocities(
    fly_node_locations, 0, fly_node_locations.shape[0]
).astype(np.float32) * (1/px_mm) * (frame_rate)

for fly_idx in tqdm(range(1, instance_count)):
    current_fly_node_locations = locations[:, :, :, [fly_idx]]
    current_fly_node_locations = trx_utils.smooth_median(current_fly_node_locations, window=5)
    current_fly_node_velocities = trx_utils.instance_node_velocities(
        current_fly_node_locations, 0, current_fly_node_locations.shape[0]
    ).astype(np.float32) * (1/px_mm) * (1/frame_rate)
    fly_node_velocities = np.dstack((fly_node_velocities, current_fly_node_velocities))
    fly_node_locations = np.concatenate((fly_node_locations, current_fly_node_locations), axis=3)


# %%
# np.save(data_dir +"/fly_node_locations_mediansmoothed.npy",fly_node_locations)
# np.save(data_dir +"/fly_node_velocities.npy",fly_node_velocities)

fly_node_locations = np.load(data_dir +"/fly_node_locations_mediansmoothed.npy")
fly_node_velocities = np.load(data_dir + "/fly_node_velocities.npy")

# %% 
fly_node_locations[:,:,1,:] = -fly_node_locations[:,:,1,:]
assignment_indices, locations, freq = trx_utils.hist_sort(
            fly_node_locations, ctr_idx=node_names.index("thorax"), ymin=-1536,ymax=0
        )
fly_node_locations[:,:,1,:] = -fly_node_locations[:,:,1,:]

# %%
import statsmodels.tsa.stattools as st
import statsmodels.graphics.tsaplots as tsaplots
tsaplots.plot_acf(data,lags=1000)
# %%

# %%
import scipy.stats
from datetime import datetime
FMT = '%w-%H:%M:%S'

time = datetime.strptime('0-22:33:00', FMT)
start_day = datetime.strptime('1-00:00:00', FMT) # for example
difference = start_day - time 
shift = int(difference.seconds*frame_rate)

plt.rcParams["figure.figsize"] = (9,3)
start_frame = 0
end_frame = int(24*60*60*frame_rate)
data = fly_node_velocities[:,node_names.index("thorax"),:]
frame_idx = (np.arange(data.shape[0]) - shift)
ToD = frame_idx % int(24*60*60*frame_rate) / (1*60*60*frame_rate)

ToD = ToD[int(6*60*60*frame_rate):]
data = data[int(6*60*60*frame_rate):,:]

plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'
# fig, ax = plt.subplots(figsize=(9, 3), dpi=300)
for fly_idx in range(data.shape[1]):
    binned = scipy.stats.binned_statistic(ToD, data[:,fly_idx],statistic='sum', bins=24, range=None)
    plt.bar(x=np.arange(24),height=binned.statistic,color=palettable.wesanderson.Moonrise7_5.mpl_colors[fly_idx],alpha=0.2,width=1)
    plt.xticks(rotation=80)
    # plt.axvspan(xmin=7.5,xmax=19.5,alpha=0.2)
    plt.tight_layout()
    plt.show()



# %%
data = fly_node_velocities[:,node_names.index("thorax"),0]
sns.histplot(data)
# %%
# plt.acorr(fly_node_velocities[0:int(1*60*60*frame_rate),node_names.index("thorax"),0],maxlags=int(1*60*frame_rate))


lags, acorrels = trx_utils.acorr(data,maxlags=int(frame_rate))
lags = lags[lags.size // 2 :]
acorrels = acorrels[acorrels.size // 2 :]
sns.lineplot(x=lags,y=acorrels)
plt.show

# %%

df = fly_node_velocities
names = ['x', 'y', 'z']
index = pd.MultiIndex.from_product([range(s)for s in df.shape], names=names)
df = pd.DataFrame({'A': df.flatten()}, index=index)['A']
df = df.unstack(level='x').swaplevel().sort_index()
df.index.names = ['fly_idx', 'node_idx']
df = pd.melt(df.reset_index(),id_vars=['node_idx','fly_idx'])
df = df.rename(columns={'x': 'frame_idx'})

# %%

# vels_temp = fly_node_velocities[0:1000,:,1]
# THORAX_INDEX
node_names_hr = ["Head", "Eye L", "Eye R", "Thorax", "Abdomen", "Foreleg L", "Foreleg R", "Midleg L", "Midleg R", "Hindleg L", "Hindleg R", "Wing L", "Wing R"]
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [3, 9.0]
plt.rcParams['figure.dpi'] = 600
plot_df = pd.DataFrame(fly_node_velocities)
plot_df = plot_df.melt(var_name="node_idx")
jp_df = plot_df[(plot_df["value"] > 0.1) & (plot_df['node_idx'] != node_names.index("proboscis"))]
fig,ax = joypy.joyplot(jp_df.sample(1000),by="node_idx",column="value",alpha=0.3,colormap=palettable.wesanderson.Zissou_5.mpl_colormap,title="Velocity",labels=node_names_hr,
                       overlap=0.75,kind="count");
plt.xlabel("Velocity (mm/s)");
plt.xlim(0,0.5)
plt.legend()
plt.show()
logger.info("Done")

# %%
plt.rcParams['figure.figsize'] = [9, 3.0]
plt.rcParams['figure.dpi'] = 600
h_df = plot_df[(plot_df["value"] > 0.001) & (plot_df['node_idx'] == THORAX_INDEX)]
sns.histplot(h_df,x="value",stat="density",binwidth=0.01)
plt.xlim(0,4)
plt.show()
# %% [markdown]
# ## Visualize thorax movement across video

# %%
head_loc = locations[:, HEAD_INDEX, :, :]
thorax_loc = locations[:, THORAX_INDEX, :, :]
abdo_loc = locations[:, ABDO_INDEX, :, :]


# %%
sns.set("notebook", "ticks", font_scale=1.2)
mpl.rcParams["figure.figsize"] = [15, 6]

# %%
plt.figure(figsize=(7, 7))
for i in tqdm(range(4)):
    plt.plot(
        thorax_loc[0:100, 0, i],
        thorax_loc[0:100, 1, i],
        "y",
        label=f"fly-{i}",
        color=palettable.wesanderson.Aquatic1_5.mpl_colors[i],
    )
# plt.legend(loc=10)

plt.xlim(0, 1536)
plt.xticks([])

plt.ylim(1536, 0)
plt.yticks([])
plt.title("Thorax tracks")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import matplotlib.colors as cols
from cv2 import absdiff, cv2

fig, ax = plt.subplots()
cmap = palettable.matplotlib.Viridis_20.get_mpl_colormap()
# cmap.set_bad("white")
ax.set_aspect("equal")


def alpha_cmap(cmap):
    my_cmap = cmap(np.arange(cmap.N))
    # Set a square root alpha.
    x = np.linspace(0, 1, cmap.N)
    my_cmap[:, -1] = x ** (0.5)
    my_cmap = cols.ListedColormap(my_cmap)

    return my_cmap


cap = cv2.VideoCapture(video_filenames[0])
res, frame = cap.read()
frame = frame[:, :, 0]
plt.imshow(frame, cmap="gray")
for fly_idx in range(4):
    # data = thorax_loc[thx_vel[fly_idx] > 1, :, fly_idx]
    data = locations[::100,:,:,fly_idx]
    x, y = data.T
    # nbins = 175
    # k = kde.gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min()-100:x.max()+100:nbins*1j, y.min()-100:y.max()+100:nbins*1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    kplot = sns.kdeplot(
        x=x,
        y=y,
        ax=ax,
        cmap=alpha_cmap(plt.cm.viridis),
        shade=True,
        shade_lowest=False,
        n_levels=50,
        # bw=.2,
        antialiased=True,
    )

    # zi[zi < 0.0000001] = np.nan

    # ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', alpha = .5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect("equal")
# ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(5, 5, True)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis("off")
fig.patch.set_visible(False)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.show()

# %%
thx_vel = [trx_utils.smooth_diff(thorax_loc[:, :, fly_idx]) for fly_idx in range(4)]

# %%
mpl.rcParams["figure.dpi"] = 600
fig, ax = plt.subplots()
for i in range(4):
    sns.histplot(
        # thx_vel[i][thx_vel[i] > 1],
        bins=100,
        stat="probability",
        binwidth=0.1,
        alpha=0.5,
        color=palettable.wesanderson.Aquatic1_5.mpl_colors[i],
    )
plt.legend(labels=["1", "2", "3", "4"])
plt.xlabel("Thorax Velocity (px)")
plt.xlim(1, 8)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.show()

# %%
thx_vel_fly = trx_utils.smooth_diff(thorax_loc[:, :, fly_idx])

# %%
thorax_x = thorax_loc[:, 0, fly_idx]
thorax_y = thorax_loc[:, 1, fly_idx]
fig, ax = plt.subplots()
ax.set_aspect("equal")
plt.plot(thorax_x[0:300000], thorax_y[0:300000])
plt.xlim(thorax_x.min(), thorax_x.max())
plt.ylim(thorax_y.min(), thorax_y.max())
plt.show()

# %%
# %matplotlib ipympl

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from cv2 import cv2
from matplotlib.animation import FuncAnimation
import palettable

# cap = cv2.VideoCapture("/Users/wolf/git/motionmapperpy/merged_25h.mp4")


# xlims = [np.nanmin(data[:,:,0]), np.nanmax(data[:,:,0])]
# ylims = [np.nanmin(data[:,:,1]), np.nanmax(data[:,:,1])]


def animate(i):
    plt.cla()
    for fly_idx in range(4):
        data = locations[25:1000, 0:13, :, fly_idx]
        data = trx_utils.smooth_median(data)
        data = trx_utils.smooth_gaussian(data)
        data[data == 0] = np.nan
        frame_numbers = list(range(1000))
        trail_length = 25
        frame_idx = frame_numbers[i] + 25
        # plt.xlim(xlims);
        # plt.ylim(ylims);
        ax.axis("off")
        ax.set_axis_off()
        fig.add_axes(ax)
        fig.patch.set_visible(False)
        print(f"Saving frame {i}")
        data_subset = data[(frame_idx - trail_length) : frame_idx]
        for node_idx in range(0, data_subset.shape[1])[:-1]:
            for idx in range(
                data_subset.shape[0]
            ):  # Note that you need to use single steps or the data has "steps"
                plt.plot(
                    data_subset[idx : (idx + 2), node_idx, 0],
                    data_subset[idx : (idx + 2), node_idx, 1],
                    linewidth=idx / trail_length,
                    color=palettable.tableau.Tableau_20.mpl_colors[node_idx],
                )
    cap.set(cv2.CAP_PROP_POS_FRAMES,25+frame_idx - 1)
    res, frame = cap.read()
    frame = frame[:, :, 0]
    plt.imshow(frame, cmap="gray")


fig, ax = plt.subplots()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(5, 5, True)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis("off")
fig.patch.set_visible(False)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

plt.rcParams['animation.ffmpeg_path'] = '/Genomics/argo/users/swwolf/.conda/envs/sleap_dev/bin/ffmpeg'
ani = FuncAnimation(fig, animate, frames=list(range(200, 205, 1)), interval=100).save(
    "withframe.mp4",
    fps=25,
    bitrate=20000,
    dpi=300,
    writer="ffmpeg",
    codec="libx264",
    extra_args=["-preset", "veryslow", "-pix_fmt", "yuv420p"]
)
plt.show()

# %%
data = locations[789950:7900550, 0:14, :, fly_idx]
data.shape

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from cv2 import cv2

cap = cv2.VideoCapture(video_filenames[0])


data = locations[:, :, :, fly_idx]
data[data == 0] = np.nan


frame_numbers = list(range(0,50, 1))

trail_length = 100
for frame_idx in tqdm([range(0, data.shape[0])[k] for k in frame_numbers]):
    fig, ax = plt.subplots()
    plt.xlim(np.nanmin(data[:, :, 0]), np.nanmax(data[:, :, 0]))
    plt.ylim(np.nanmin(data[:, :, 1]), np.nanmax(data[:, :, 1]))
    data_subset = data[(frame_idx - trail_length) : frame_idx]
    for node_idx in range(0, data_subset.shape[1])[:-1]:
        for idx in range(data_subset.shape[0]):
            plt.plot(
                data_subset[idx : (idx + 2), node_idx, 0],
                data_subset[idx : (idx + 2), node_idx, 1],
                linewidth=idx / data_subset.shape[0],
                color=palettable.tableau.Tableau_20.mpl_colors[node_idx],
            )
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    res, frame = cap.read()
    frame = frame[:, :, 0]
    plt.imshow(frame, cmap="gray")
    # plt.show()

# def animate(i):


# for i in tqdm(range(0,data.shape[1])[:-1]): # Ignore proboscis
#     frames = [range(0,data.shape[0])[k] for k in frame_numbers]
#     for j in range(len(frames)-1):
#         data_subset = data[frames[(j-trail_length):j]]
#         for k in range(data_subset.shape[0]):
#             plt.plot(data_subset[k:(k+2),i,0],data_subset[k:(k+2),i,1], linewidth = 1,color=palettable.tableau.Tableau_20.mpl_colors[i])
# ax.set_aspect('equal');

# plt.show()

# %%
# import logging

# logger = logging.getLogger("analysis_logger")

# filename = "/Volumes/Seagate Backup  1/lts/Exp2_Camera1_25h.analysis.slp"

# %% [markdown]
# ## More advanced visualizations
#
# For some additional analysis, we'll first smooth and differentiate the data with a Savitzky-Golay filter to extract velocities of each joint.

# %%
# thx_vel_fly = smooth_diff(thorax_loc[:, :, fly_idx])

# %% [markdown]
# ### Visualizing thorax x-y dynamics and velocity for fly 0

# %%
fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(211)
ax1.plot(thorax_loc[:, 0, 0], "k", label="x")
ax1.plot(-1 * thorax_loc[:, 1, 0], "k", label="y")
ax1.legend()
ax1.set_xticks([])
ax1.set_title("Thorax")

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(thx_vel_fly0[:, np.newaxis].T, aspect="auto", vmin=0, vmax=10)
ax2.set_yticks([])
ax2.set_title("Velocity")

# %% [markdown]
# ### Visualize thorax colored by magnitude of fly speed

# %%
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ax1.plot(thorax_loc[:, 0, 0], thorax_loc[:, 1, 0], "k")
ax1.set_xlim(0, 1024)
ax1.set_xticks([])
ax1.set_ylim(0, 1024)
ax1.set_yticks([])
ax1.set_title("Thorax tracks")

kp = thx_vel_fly0  # use thx_vel_fly1 for other fly
vmin = 0
vmax = 10

ax2 = fig.add_subplot(122)
ax2.scatter(thorax_loc[:, 0, 0], thorax_loc[:, 1, 0], c=kp, s=4, vmin=vmin, vmax=vmax)
ax2.set_xlim(0, 1024)
ax2.set_xticks([])
ax2.set_ylim(0, 1024)
ax2.set_yticks([])
ax2.set_title("Thorax tracks colored by magnitude of fly speed")

# %% [markdown]
# ## Clustering
#
# For an example of clustering the data, we'll
#
# 1. extract joint velocities for each joint,
# 2. run simple k-means on the velocities from each frame.
#

# %%
def instance_node_velocities(fly_node_locations, start_frame, end_frame, frame_count):
    fly_node_velocities = np.zeros((frame_count, 13))

    for n in range(0, node_count):
        fly_node_velocities[:, n] = smooth_diff(
            fly_node_locations[start_frame:end_frame, n, :]
        )

    return fly_node_velocities


# %%
def instance_node_velocities(fly_node_locations, start_frame, end_frame, frame_count):
    fly_node_velocities = np.zeros((frame_count, 13))

    for n in range(0, node_count):
        fly_node_velocities[:, n] = smooth_diff(
            fly_node_locations[start_frame:end_frame, n, :]
        )

    return fly_node_velocities


# %%
def plot_instance_node_velocities(instance_idx, node_velocities):
    plt.figure(figsize=(20, 8))
    plt.imshow(
        node_velocities.T, aspect="auto", vmin=0, vmax=20, interpolation="nearest"
    )
    plt.xlabel("frames")
    plt.ylabel("nodes")
    plt.yticks(np.arange(node_count), node_names, rotation=20)
    plt.title(f"Fly {instance_idx} node velocities")


# %%
fly_idx = 0
fly_node_velocities = instance_node_velocities(fly_idx)

# %%
plot_instance_node_velocities(fly_idx, fly_node_velocities[7900200:7900400, :])
plt.show()

# %%
# day = np.concatenate([fly_node_velocities[0:(34020*100),node_names.index('thorax')],fly_node_velocities[(34020*100 + 28800*100):(34020*100 + 28800*100 + 28800*100),node_names.index('thorax')]])

# night = fly_node_velocities[(34020*100):(34020*100 + 28800*100),node_names.index('thorax')]
day = np.concatenate(
    [
        range(34020 * 100),
        range((34020 * 100 + 28800 * 100), (34020 * 100 + 28800 * 100 + 28800 * 100)),
    ]
)
night = np.array(range((34020 * 100), (34020 * 100 + 28800 * 100)))
mpl.rcParams["figure.dpi"] = 600
fig, ax = plt.subplots()
# for i in range(4):
sns.histplot(
    np.concatenate([thx_vel[i][day[day < 8999999]] for i in range(4)])[::100],
    stat="probability",
    binwidth=0.005,
    alpha=0.5,
    color=palettable.wesanderson.Aquatic1_5.mpl_colors[0],
)

sns.histplot(
    np.concatenate([thx_vel[i][night[night < 8999999]] for i in range(4)])[::100],
    stat="probability",
    binwidth=0.005,
    alpha=0.5,
    color=palettable.wesanderson.Aquatic1_5.mpl_colors[1],
)

plt.legend(labels=["1", "2", "3", "4"])
# plt.xlabel("Thorax Velocity (px)")
plt.xlim(0, 0.01)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.show()


# %%
day_vel = [thx_vel[i][day[day < 8999999]] for i in range(4)]
night_vel = [thx_vel[i][night[night < 8999999]] for i in range(4)]


day_vel2 = [thx_vel[i][day[day < 8999999]] for i in range(4)]
night_vel2 = [thx_vel[i][night[night < 8999999]] for i in range(4)]

print(len(day_vel2[0]))
print(len(night_vel2[0]))

day_vel_lens = [len(day_vel[day_vel > 1]) / len(day_vel2[0]) for day_vel in day_vel]
night_vel_lens = [
    len(night_vel[night_vel > 1]) / len(night_vel2[0]) for night_vel in night_vel
]

# %%
fig, ax = plt.subplots()
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
fig.add_axes(ax)
fig.set_size_inches(5, 5, True)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax.axis('off')
fig.patch.set_visible(False)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

sns.boxplot(
    ["Day", "Day", "Day", "Day", "Night", "Night", "Night", "Night"],
    day_vel_lens + night_vel_lens,
    palette=palettable.wesanderson.Aquatic2_5.mpl_colors,
)
# sns.swarmplot(["Day","Day","Day","Day","Night","Night","Night","Night"], day_vel_lens + night_vel_lens)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.ylabel("Fraction of time spent moving")
plt.show()


# %%
d = np.concatenate([thx_vel[i][day[day < 9000000]] for i in range(4)])

# %%
np.max(day)

# %%
sns.kdeplot(data=[day[::1000], night[::1000]], bw=0.1)
plt.show()

# %%
day = fly_node_velocities[0 : 34020 * 100, node_names.index("thorax")]
day.shape


# %%
from sklearn.cluster import KMeans

# %%
nstates = 10

km = KMeans(n_clusters=nstates)

labels = km.fit_predict(fly_node_velocities)

# %%
fig = plt.figure(figsize=(20, 12))

ax1 = fig.add_subplot(211)
ax1.imshow(
    fly_node_velocities.T, aspect="auto", vmin=0, vmax=20, interpolation="nearest"
)
ax1.set_xlabel("Frames")
ax1.set_ylabel("Nodes")
ax1.set_yticks(np.arange(node_count))
ax1.set_yticklabels(node_names)
ax1.set_title(f"Fly {fly_idx} node velocities")
ax1.set_xlim(0, frame_count)

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(labels[None, :], aspect="auto", cmap="tab10", interpolation="nearest")
ax2.set_xlabel("Frames")
ax2.set_yticks([])
ax2.set_title("Ethogram (colors = clusters)")

# %% [markdown]
# ## More advanced visualizations
#
# For some additional analysis, we'll first smooth and differentiate the data with a Savitzky-Golay filter to extract velocities of each joint.

# %% [markdown]
# ### Visualizing thorax x-y dynamics and velocity for fly 0

# %%
fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(211)
ax1.plot(thorax_loc[:, 0, 0], "k", label="x")
ax1.plot(-1 * thorax_loc[:, 1, 0], "k", label="y")
ax1.legend()
ax1.set_xticks([])
ax1.set_title("Thorax")

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(thx_vel_fly0[:, np.newaxis].T, aspect="auto", vmin=0, vmax=10)
ax2.set_yticks([])
ax2.set_title("Velocity")

# %% [markdown]
# ### Visualize thorax colored by magnitude of fly speed

# %%
fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ax1.plot(thorax_loc[:, 0, 0], thorax_loc[:, 1, 0], "k")
ax1.set_xlim(0, 1024)
ax1.set_xticks([])
ax1.set_ylim(0, 1024)
ax1.set_yticks([])
ax1.set_title("Thorax tracks")

kp = thx_vel_fly0  # use thx_vel_fly1 for other fly
vmin = 0
vmax = 10

ax2 = fig.add_subplot(122)
ax2.scatter(thorax_loc[:, 0, 0], thorax_loc[:, 1, 0], c=kp, s=4, vmin=vmin, vmax=vmax)
ax2.set_xlim(0, 1024)
ax2.set_xticks([])
ax2.set_ylim(0, 1024)
ax2.set_yticks([])
ax2.set_title("Thorax tracks colored by magnitude of fly speed")

# %% [markdown]
# ### Find covariance in thorax velocities between fly-0 and fly-1

# %%
import pandas as pd





# %%
win = 50

cov_vel = corr_roll(thx_vel_fly0, thx_vel_fly1, win)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
ax[0].plot(thx_vel_fly0, "y", label="fly-0")
ax[0].plot(thx_vel_fly1, "g", label="fly-1")
ax[0].legend()
ax[0].set_title("Forward Velocity")

ax[1].plot(cov_vel, "c", markersize=1)
ax[1].set_ylim(-1.2, 1.2)
ax[1].set_title("Covariance")

fig.tight_layout()

# %% [markdown]
# ## Clustering
#
# For an example of clustering the data, we'll
#
# 1. extract joint velocities for each joint,
# 2. run simple k-means on the velocities from each frame.

# %%
fly_idx = 0
fly_node_velocities = instance_node_velocities(fly_idx)
plot_instance_node_velocities(fly_idx, fly_node_velocities)

# %%
from sklearn.cluster import KMeans

# %%
nstates = 10

km = KMeans(n_clusters=nstates)

labels = km.fit_predict(fly_node_velocities)

# %%
fig = plt.figure(figsize=(20, 12))

ax1 = fig.add_subplot(211)
ax1.imshow(
    fly_node_velocities.T, aspect="auto", vmin=0, vmax=20, interpolation="nearest"
)
ax1.set_xlabel("Frames")
ax1.set_ylabel("Nodes")
ax1.set_yticks(np.arange(node_count))
ax1.set_yticklabels(node_names)
ax1.set_title(f"Fly {fly_idx} node velocities")
ax1.set_xlim(0, frame_count)

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(labels[None, :], aspect="auto", cmap="tab10", interpolation="nearest")
ax2.set_xlabel("Frames")
ax2.set_yticks([])
ax2.set_title("Ethogram (colors = clusters)")