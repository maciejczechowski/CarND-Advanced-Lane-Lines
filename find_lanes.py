import glob
import os
import cv2
import numpy
from src import pipeline
from src import camera
from src import parameters
import argparse

from moviepy.editor import VideoFileClip


def process_frame(image):
    process.frame += 1
    res, lanes = pipeline.process_image(image, params, process)
    return res


params = parameters.LaneFinderParams()
process = parameters.LaneFinderProcess()



process = parameters.LaneFinderProcess()

parser = argparse.ArgumentParser(description='Visualizes the threshold process.')
parser.add_argument('filename')
parser.add_argument('output')

args = parser.parse_args()
movie = args.filename

clip2 = VideoFileClip(movie)#.subclip(23,27) #.subclip(39,42)
result_clip = clip2.fl_image(process_frame)
result_clip.write_videofile(args.output, audio=False, logger="bar")

