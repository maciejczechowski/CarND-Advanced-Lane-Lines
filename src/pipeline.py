from . import camera
from . import lane_finder as lf
from . import renderer
from . import sanitizer
from . import parameters
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def process_image(image, params):
    image_undistorted = camera.undistort(image, params.camera_mtx, params.camera_dist)
    image_thresholded = lf.threshold(image_undistorted, params.thresh_s, params.thresh_sx, params.thresh_h)
    image_warped = lf.toBirdsEye(image_thresholded, params.warp_x1, params.warp_x2, params.warp_horizon)

    leftx, lefty, lfound, rightx, righty, rfound, out_img = lf.find_lane_pixels_v2(image_warped,
                                                                                   params.lane_nwindows,
                                                                                   params.lane_margin,
                                                                                   params.lane_minpix)
    left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr = lf.fit_poly(image.shape, leftx, lefty, rightx, righty)

    # sanitize #found
    min_found = params.lane_nwindows / 3
  #  if lfound >= min_found and rfound < min_found:  # left lane okay, right broken. take c from right and a,b from left
  #      print("extrapolating")
   #     right_fit_cr, right_fitx = lf.extrapolate_fit(image.shape, left_fit_cr, right_fit_cr)


    image_lines = renderer.draw_lanes(np.zeros_like(out_img),
                                      left_fitx,
                                      right_fitx,
                                      ploty)
    # out_img = renderer.draw_lanes(out_img,
    #                                   left_fitx,
    #                                   right_fitx,
    #                                   ploty)

    lanes_unwarped = lf.fromBirdsEye(image_lines, params.warp_x1, params.warp_x2, params.warp_horizon)
    result = renderer.weighted_img(lanes_unwarped, image)

    # SANITIZER
    left_curvature, right_curvature = sanitizer.measure_curvature_real(left_fitx, right_fitx, ploty)
    position = sanitizer.calulate_position(left_fitx, right_fitx)
    lane_width_mean, lane_width_min, lane_width_max, avg_diff = sanitizer.calulate_lane_size(left_fitx, right_fitx)
    sanity = sanitizer.check_lanes(left_fit_cr, right_fit_cr, ploty)

    renderer.addText(result, ["LC: " + str(left_curvature),
                              "RC: " + str(right_curvature),
                              "position: " + str(position),
                              "lane_width: " + str(lane_width_mean),
                              "lane max " + str(lane_width_max),
                              "lane min" + str(lane_width_min),
                              "aD" + str(np.abs(left_fit_cr[0] - right_fit_cr[0])),
                              "bD" + str(np.abs(left_fit_cr[1] - right_fit_cr[1])),
                              "sanity " + str(sanity)
                              ])
    wimg = np.dstack((image_warped * 255, image_warped, image_warped))
    r1 = renderer.weighted_img(out_img, wimg)
    return result, out_img
