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
    res, lanes = pipeline.process_image(image, params)
    return res



#calibration_image_files = glob.glob("camera_cal/calibration*.jpg")
#ret, mtx, dist, rvecs, tvecs = camera.calibrate_images(calibration_image_files, 9, 6)

params = parameters.LaneFinderParams()
#parameters.camera_mtx = mtx
#parameters.camera_dist = dist

process = parameters.LaneFinderProcess()



movie = "./project_video.mp4"
movie = "./challenge_video.mp4"
clip2 = VideoFileClip(movie)#.subclip(39,42)
result_clip = clip2.fl_image(process_frame)
result_clip.write_videofile("result.mp4", audio=False, logger="bar")


# image = cv2.imread("test-fr/p3.jpg")
# final = pipeline.process_image(image, params)
#



#
# testFiles = os.listdir("test_images")
# for testFile in testFiles:
#     image = cv2.imread("test_images/" + testFile)
#     final = pipeline.process_image(image, params)
#
#     cv2.imwrite("output_test/lanes_" + testFile, final)

