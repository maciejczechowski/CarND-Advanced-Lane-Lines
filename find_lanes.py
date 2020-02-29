import glob
import os
import cv2
import numpy
from src import pipeline
from src import camera
from src import parameters

from moviepy.editor import VideoFileClip


def process_frame(image):
    process.frame += 1
    res = pipeline.process_image(image, params)
    return res



#calibration_image_files = glob.glob("camera_cal/calibration*.jpg")
#ret, mtx, dist, rvecs, tvecs = camera.calibrate_images(calibration_image_files, 9, 6)

params = parameters.LaneFinderParams()
#parameters.camera_mtx = mtx
#parameters.camera_dist = dist

process = parameters.LaneFinderProcess()


## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)

movie = "./project_video.mp4"
clip2 = VideoFileClip(movie)
result_clip = clip2.fl_image(process_frame)
result_clip.write_videofile("result.mp4", audio=False, logger="bar")



#
# testFiles = os.listdir("test_images")
# for testFile in testFiles:
#     image = cv2.imread("test_images/" + testFile)
#     final = pipeline.process_image(image, params)
#
#     cv2.imwrite("output_test/lanes_" + testFile, final)

