import os
import argparse
import cv2

'''
This script will export frames from a video using ffmpeg (needs to be installed and added to PATH).
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-f', '--fps', type=float, default=None)
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-e', '--end', type=int, default=1)
    parser.add_argument('--sequence', action='store_true')

    opt = parser.parse_args()
    video = cv2.VideoCapture(opt.input)
    fps = video.get(cv2.CAP_PROP_FPS) if opt.fps is None else opt.fps
    start = opt.start - 1
    lenght = opt.end - start - 1
    if opt.sequence:
        if os.path.isfile(opt.output):
            outdir = os.path.split(opt.output)[0]
        else:
            outdir = opt.output
        cmd = f'ffmpeg -i "{opt.input}" -ss {start/fps} -r {fps} -frames:v {int(lenght)} -start_number {opt.start} {outdir}\%04d.png'
    else:
        cmd = f'ffmpeg -i "{opt.input}" -ss {start/fps} -c:v libx264 -r {fps} -frames:v {int(lenght)} {opt.output}'
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    main()
