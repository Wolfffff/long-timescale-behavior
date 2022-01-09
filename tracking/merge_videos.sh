# Generate a list of files to merge -- we truncate to the last full hour here.
# Set g anre remove the unset to slice into groups of g files.
FILES=(INFO*.mp4)
g=999
IFS=$'\n'

for((i=0; i < ${#FILES[@]}; i+=g))
do
  part=( "${FILES[@]:i:g}" )
  unset 'part[${#part[@]}-1]'
  echo "Elements in this group: ${part[*]}"
  printf "file '%s'\n" "${part[@]}" > ${i}through$((${i}+${#part[@]}-1)).txt
done

# FFmpeg fails if we choose mp4 as the output format so just use mkv...
ffmpeg -f concat -safe 0 -i 0throughN -c copy 0throughN.mkv

# Clean up the files we created
python ../sleap/sleap/nn/tracking.py ${FILE}.slp --tracker simple --similarity iou --match greedy --track_window 3 --post_connect_single_breaks 1 --target_instance_count 4 --clean_instance_count 4 --output ${FILE}.tracked.slp

#
# for FILE in *through*.slp; do $(python ../../sleap/sleap/nn/tracking.py ${FILE} --tracker simple --similarity iou --match greedy --track_window 5 --post_connect_single_breaks 1 --target_instance_count 4 --clean_instance_count 4 --output ${FILE%.*}.tracked.slp) & ; done