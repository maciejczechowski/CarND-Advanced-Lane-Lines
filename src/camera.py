import numpy as np
import cv2

def calibrate_images(images, nx, ny):
    objpoints = []  # 3d points from real world
    imgpoints = []  # 2d points on image

    #prepare object points  - points of cheessboard in undistoretd space

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    size = (0, 0)
    for imageFile in images:
        image = cv2.imread(imageFile)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    return  ret, mtx, dist, rvecs, tvecs

def undistort(image, mtx, dist):
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    return dst

def warp(image, src, dst):
    size = image.shape
    w = size[1]
    h = size[0]
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped




