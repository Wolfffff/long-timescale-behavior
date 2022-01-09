import glob
import re
from time import time
import sleap
from tqdm import tqdm
import h5py
import rich
import argparse
import cv2
import os.path

parser = argparse.ArgumentParser(description="generate frame slices")
parser.add_argument("target", type=str, help="target", nargs="+")
args = parser.parse_args()

print(args.target)

filenames = glob.glob(args.target[0] + "*.slp")
filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
print(filenames)
size = 24
groups = [filenames[i:i+24] for i in range(0, len(filenames), 24)]
print(groups)
for i in range(len(groups)):
    group = groups[i]
    base_labels = sleap.Labels.load_file(group[0])
    for filename in group[1::]:
        new_labels = sleap.Labels.load_file(filename,match_to=base_labels)
        base_labels.extend_from(new_labels)
    base_labels.save_file(f'{args.target[0]}_{(i*size) + 1}through{i*size}.slp')