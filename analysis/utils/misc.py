#%%
# for f in *.slp; do sleap-convert --format analysis ${f}; done

import numpy as np
import sleap
import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
import re

filenames = glob(
    "/Users/wolf/git/long-timescale-behavior/tracking/sleap/20210728/predictions/*.h5"
)
filenames.sort(key=lambda f: int(re.sub("\\D", "", f)))
print(filenames)


def hist_sort(locations, xbins=2, ybins=2, xmin=0, xmax=1536, ymin=0, ymax=1536):
    assignments = []
    for track_num in range(locations.shape[3]):
        h, xedges, yedges = np.histogram2d(
            locations[:, node_names.index("thorax"), 0, track_num],
            locations[:, node_names.index("thorax"), 1, track_num],
            range=[[xmin, xmax], [ymin, ymax]],
            bins=[xbins, ybins],
        )
        # Probably not the correct way to flip this around to get rows and columns correct but it'll do!
        assignment = h.argmax()
        assignments.append(assignment)
    assignment_indices = np.argsort(assignments)
    locations = locations[:, :, :, assignment_indices]
    return assignment_indices, locations


for filename in filenames:
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        node_names = [n.decode() for n in f["node_names"][:]]
        occupancy_current = f["track_occupancy"][:]
        first_nonzero_location = np.min((occupancy_current != 0).argmax(axis=0)[0])
        occupancy_current = occupancy_current[first_nonzero_location::, :]
        print(occupancy_current.shape)
        locations_current = f["tracks"][:].T
        locations_current = locations_current[first_nonzero_location::, :, :, :]
        locations_current[:, :, 1, :] = -locations_current[:, :, 1, :]
        assignment_indices, locations_current = hist_sort(locations_current)
        occupancy_current = occupancy_current[:, assignment_indices]
        print(assignment_indices)
        # print(locations_current.shape)
    try:
        locations = np.vstack((locations, locations_current))
        occupancy = np.vstack((occupancy, occupancy_current))
    except:
        locations = locations_current
        occupancy = occupancy_current
print(locations.shape)
print(occupancy.shape)

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

# 55
frame_count, node_count, _, instance_count = locations.shape
node_name = "thorax"
start_frame = 0
end_frame = frame_count
track_num = range(0, 4)
x = locations[
    start_frame:end_frame:100, node_names.index(node_name), 0, track_num
].flatten()
y = locations[
    start_frame:end_frame:100, node_names.index(node_name), 1, track_num
].flatten()
sns.jointplot(x=x, y=y, kind="hex")
plt.show()
# %%
