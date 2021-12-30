import glob
import re
import time
import sleap

filenames = glob.glob(
    "*.slp"
)
filenames.sort(key=lambda f: int(re.sub("\D", "", f)))
print(filenames)

t1_start = time.process_time()
base_labels = sleap.Labels.load_file(filenames[0])

for filename in filenames[1::]:
    new_labels = sleap.Labels.load_file(filename)
    base_labels = sleap.Labels.complex_merge_between(
        base_labels, new_labels, unify=True
    )

t1_stop = time.process_time()

print("Elapsed time:", t1_stop, t1_start)
sleap.Labels.save_file(base_labels, "merged.slp")
