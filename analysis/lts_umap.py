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
import utils.trx_utils as trx_utils


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis_logger")
filenames = ["data/20211112_1900_h265_23_05mgmlCaf_2021-11-17_cam1.tracked.analysis.h5","data/20211112_1900_h265_23_05mgmlCaf_2021-11-17_cam2.tracked.analysis.h5"]
video_filenames = ["data/videos/0through23_cam1.mp4","data/videos/0through23_cam2.mp4"]

frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px

# %%

with h5py.File(filenames[0], "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]
    locations[:,:,1,:] = -locations[:,:,1,:]
    assignment_indices, locations = trx_utils.hist_sort(
        locations, ctr_idx=node_names.index("thorax")
    )
    locations[:,:,1,:] = -locations[:,:,1,:]

for filename in filenames[1:]:
    with h5py.File(filename, "r") as f:
        temp_locations = f["tracks"][:].T
        temp_locations[:,:,1,:] = -temp_locations[:,:,1,:]
        temp_assignment_indices, temp_locations = trx_utils.hist_sort(
            temp_locations, ctr_idx=node_names.index("thorax")
        )
        temp_locations[:,:,1,:] = -temp_locations[:,:,1,:]

        locations = np.concatenate((locations, temp_locations), axis=3)

locations = np.delete(locations, node_names.index("proboscis"), axis=1)
locations = trx_utils.fill_missing_np(locations)


print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")

# %%
frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

HEAD_INDEX = node_names.index("head")
THORAX_INDEX = node_names.index("thorax")
ABDO_INDEX = node_names.index("abdomen")

# %%
import glob, os, pickle
from datetime import datetime
import numpy as np
from scipy.io import loadmat, savemat
import hdf5storage
import utils.motionmapperpy.motionmapperpy as mmpy

projectPath = "gxe"
mmpy.createProjectDirectory(projectPath)

for i in range(instance_count):
    data = locations[
        : int(24 * 60 * 60 * frame_rate), :, :, i
    ]  # 0:int(24*60*60*frame_rate)
    data = trx_utils.smooth_median(data, window=5)
    vels = trx_utils.instance_node_velocities(data, 0, data.shape[0]).astype(np.float32)
    data = trx_utils.normalize_to_egocentric(
        x=data, ctr_ind=THORAX_INDEX, fwd_ind=HEAD_INDEX
    )
    # mask = (vels > 3).any(axis=1)
    # data = data[mask,:]
    # data = data[np.random.randint(0,data.shape[0],size=10000),:]
    data = data.reshape((data.shape[0], 2 * data.shape[1]))
    print(data.shape)
    savemat(
        projectPath + "/Projections/dataset_%i_pcaModes.mat" % (i + 1),
        {"projections": data},
    )

parameters = mmpy.setRunParameters()

# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.projectPath = projectPath
parameters.method = "UMAP"

parameters.waveletDecomp = (
    True  #% Whether to do wavelet decomposition. If False, PCA projections are used for
)
#% tSNE embedding.

parameters.minF = 0.5  #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = frame_rate / 2  #% Maximum frequency for Morlet Wavelet Transform,
#% equal to Nyquist frequency for your measurements.

# parameters.perplexity = 5
# parameters.training_perplexity = 20

parameters.samplingFreq = frame_rate  #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 25  #% No. of frequencies between minF and maxF.

parameters.numProcessors = -1  #% No. of processor to use when parallel
#% processing (for wavelets, if not using GPU). -1 to use all cores.

parameters.useGPU = -1  # GPU to use, set to -1 if GPU not present

parameters.training_numPoints = int(
    1 * 60 * 60 * frame_rate
)  #% Number of points in mini-tSNEs.

# %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
parameters.trainingSetSize = 100000  #% Total number of representative points to find. Increase or decrease based on
#% available RAM. For reference, 36k is a good number with 64GB RAM.

parameters.embedding_batchSize = (
    3000  #% Lower this if you get a memory error when re-embedding points on learned
)
#% tSNE map.


projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")

m = loadmat(projectionFiles[0], variable_names=["projections"])["projections"]

# %%%%%
parameters.pcaModes = m.shape[1]  #%Number of PCA projections in saved files.
parameters.numProjections = parameters.pcaModes
# %%%%%
del m

print(datetime.now().strftime("%m-%d-%Y_%H-%M"))
print("tsneStarted")

if parameters.method == "TSNE":
    if parameters.waveletDecomp:
        tsnefolder = parameters.projectPath + "/TSNE/"
    else:
        tsnefolder = parameters.projectPath + "/TSNE_Projections/"
elif parameters.method == "UMAP":
    tsnefolder = parameters.projectPath + "/UMAP/"

if not os.path.exists(tsnefolder + "training_tsne_embedding.mat"):
    print("Running minitSNE")
    mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
    print("minitSNE done, finding embeddings now.")
    print(datetime.now().strftime("%m-%d-%Y_%H-%M"))

import h5py

with h5py.File(tsnefolder + "training_data.mat", "r") as hfile:
    trainingSetData = hfile["trainingSetData"][:].T


with h5py.File(tsnefolder + "training_embedding.mat", "r") as hfile:
    trainingEmbedding = hfile["trainingEmbedding"][:].T


if parameters.method == "TSNE":
    zValstr = "zVals" if parameters.waveletDecomp else "zValsProjs"
else:
    zValstr = "uVals"


for i in range(len(projectionFiles)):
    print("Finding Embeddings")
    print("%i/%i : %s" % (i + 1, len(projectionFiles), projectionFiles[i]))
    if os.path.exists(projectionFiles[i][:-4] + "_%s.mat" % (zValstr)):
        print("Already done. Skipping.\n")
        continue
    projections = loadmat(projectionFiles[i])["projections"]
    zValues, outputStatistics = mmpy.findEmbeddings(
        projections, trainingSetData, trainingEmbedding, parameters
    )
    hdf5storage.write(
        data={"zValues": zValues},
        path="/",
        truncate_existing=True,
        filename=projectionFiles[i][:-4] + "_%s.mat" % (zValstr),
        store_python_metadata=False,
        matlab_compatible=True,
    )
    with open(
        projectionFiles[i][:-4] + "_%s_outputStatistics.pkl" % (zValstr), "wb"
    ) as hfile:
        pickle.dump(outputStatistics, hfile)
    print("Embeddings saved.\n")
    del zValues, projections, outputStatistics




print("All Embeddings Saved!")


mmpy.findWatershedRegions(
    parameters,
    minimum_regions=8,
    startsigma=0.3,
    pThreshold=[0.33, 0.67],
    saveplot=True,
    endident="*_pcaModes.mat",
)