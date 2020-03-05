from src import camera
import glob
import cv2
import os

calibration_image_files = glob.glob("camera_cal/calibration*.jpg")
ret, mtx, dist, rvecs, tvecs = camera.calibrate_images(calibration_image_files, 9, 6)

print(mtx)
print(dist)

testFiles = os.listdir("camera_cal")
for testFile in testFiles:
    image = cv2.imread("camera_cal/" + testFile)
    final = camera.undistort(image, mtx, dist)
    cv2.imwrite("calibrated_images/" + testFile, final)

