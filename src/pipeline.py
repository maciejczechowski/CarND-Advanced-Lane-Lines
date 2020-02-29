from . import camera
from . import lane_finder as lf
from . import renderer
from . import parameters
import glob
import os
import numpy as np
import cv2

def process_image(image, params):

    image_undistorted = camera.undistort(image, params.camera_mtx, params.camera_dist)
    image_thresholded = lf.threshold(image_undistorted, params.thresh_s, params.thresh_sx)
    image_warped = lf.toBirdsEye(image_thresholded, params.warp_x1, params.warp_x2, params.warp_horizon)

    leftx, lefty, rightx, righty, out_img = lf.find_lane_pixels(image_warped,
                                                                params.lane_nwindows,
                                                                params.lane_margin,
                                                                params.lane_minpix)
    left_fitx, right_fitx, ploty = lf.fit_poly(image.shape, leftx, lefty, rightx, righty)


    image_lines = renderer.draw_lanes(np.zeros_like(out_img),
                                      left_fitx,
                                      right_fitx,
                                      ploty)

    lanes_unwarped = lf.fromBirdsEye(image_lines, params.warp_x1, params.warp_x2, params.warp_horizon)
    result = renderer.weighted_img(lanes_unwarped, image)

    wimg = np.dstack((image_warped * 255, image_warped, image_warped))
    r1 = renderer.weighted_img(out_img, wimg)
    return result


