import argparse
import cv2 
import os.path

parser = argparse.ArgumentParser(description='generate frame slices')


parser.add_argument('video', type=str, help='video file', nargs='?')
parser.add_argument('number_of_slices', type=int,default=20, nargs='?', help='number of slices')

args = parser.parse_args()

def ranges(N, nb):
    step = N / nb
    return ["{}-{}".format(round(step*i), round(step*(i+1)-1)) for i in range(nb)]


video = cv2.VideoCapture(args.video)
total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_slices = ranges(total,args.number_of_slices)

base_name = os.path.basename(args.video)
no_suffix = os.path.splitext(base_name)[0]
with open(no_suffix + '.frame_list', 'w') as f:
    for item in frame_slices:
        f.write("%s\n" % item)