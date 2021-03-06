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
print(args.target[0] + ".slp")


class RateColumn(rich.progress.ProgressColumn):
    """Renders the progress rate."""

    def render(self, task: "Task") -> rich.progress.Text:
        """Show progress rate."""
        speed = task.speed
        if speed is None:
            return rich.progress.Text("?", style="progress.data.speed")
        return rich.progress.Text(f"{speed:.2f} it/s", style="progress.data.speed")


print(args.target)
filenames = glob.glob(args.target[0] + "*.slp")
filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
print(filenames)

base_labels = sleap.Labels.load_file(filenames[0])

report_period = 0.1
with rich.progress.Progress(
    "{task.description}",
    rich.progress.BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "ETA:",
    rich.progress.TimeRemainingColumn(),
    RateColumn(),
    auto_refresh=False,
    refresh_per_second=5,
    speed_estimate_period=3,
) as progress:
    task = progress.add_task("Running...", total=len(filenames[1::]))
    last_report = time()
    for filename in filenames[1::]:
        progress.update(task, advance=1)
        elapsed_since_last_report = time() - last_report
        if elapsed_since_last_report > report_period:
            progress.refresh()
        # This is the "fix"
        new_labels = sleap.Labels.load_file(filename,match_to=base_labels)
        base_labels.extend_from(new_labels)


# base_labels = sleap.Labels.complex_merge_between(base_labels, new_labels.labeled_frames)
sleap.Labels.save_file(base_labels, args.target[0] + ".slp")

#import h5py

#f = h5py.File(args.target[0] + ".slp", "r+")
#print(len(f["frames"]["video"][:]))
#print(f["frames"]["video"][:])
#f["frames"]["video"] = 0
#print(len(f["frames"]["video"][:]))
#print(f["frames"]["video"][:])
#f.close()
