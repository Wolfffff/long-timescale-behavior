#!/bin/bash

# This script generates list files containing 24 files (24h) for each day.
# srun --job-name "InteractiveJob" --cpus-per-task 24 --mem 64G --time 1-00:00:00 --pty bash

FILES=(INFO*.mp4)
g=24
IFS=$'\n'

for((i=0; i < ${#FILES[@]}; i+=g))
do
  part=( "${FILES[@]:i:g}" )
  echo "Elements in this group: ${part[*]}"
  printf "file '%s'\n" "${part[@]}" > ${i}through$((${i}+${#part[@]}-1)).txt
done

# ffmpeg -f concat -safe 0 -i 0through23.list -c copy 0through23.mp4
# {x%%.*}

#for i in $(*through*.txt ); do echo item: $i; done


# for i in *through*.txt; do echo ${i%.*}.mp4 ; (ffmpeg -f concat -safe 0 -i $i -c copy ${i%.*}.mp4 &) done &

#python ../sleap/sleap/nn/tracking.py ${FILE}.slp --tracker simple --similarity iou --match greedy --track_window 3 --post_connect_single_breaks 1 --target_instance_count 4 --clean_instance_count 4 --output ${FILE}.tracked.slp

for i in exp2*; do echo ${i%.*}.mp4 ; (ffmpeg -f concat -safe 0 -i $i -c copy ${i%.*}.mp4 &) done &
