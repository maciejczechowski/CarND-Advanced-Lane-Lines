from . import camera
from . import lane_finder as lf
from . import renderer
from . import sanitizer
from . import parameters
from . import calculations as cal
from scipy import interpolate
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from .parameters import LaneFinderProcess


def limit_change(current_cr, prev_cr, ploty):
    b_diff = np.clip(current_cr[1] - prev_cr[1], -0.5, 0.5)
    c_diff = np.clip(current_cr[2] - prev_cr[2], -30, 30)
    a = current_cr[0]
    b = prev_cr[1] + b_diff
    c = prev_cr[2] + c_diff

    fit_cr = [a, b, c]
    fitx = fit_cr[0] * ploty ** 2 + fit_cr[1] * ploty + fit_cr[2]

    return fit_cr, fitx


def process_image(image, params, process: LaneFinderProcess):
    image = camera.undistort(image, params.camera_mtx, params.camera_dist)
    image_thresholded = lf.threshold(image, params.thresh_s, params.thresh_sx, params.thresh_h)
    image_warped = lf.toBirdsEye(image_thresholded, params.warp_x1, params.warp_horizon)

    leftx, lefty, lfound, rightx, righty, rfound, out_img = lf.find_lane_pixels_v2(image_warped,
                                                                                   params.lane_nwindows,
                                                                                   params.lane_margin,
                                                                                   params.lane_minpix)

    left_fitx = None
    right_fitx = None
    left_fit_cr = None
    right_fit_cr = None

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    if (len(leftx) == 0 or len(rightx) == 0) and len(process.right_crs) < 2:  # fatal, skip frame
        return image, out_img

    if len(leftx) > 0:
        left_fitx, left_fit_cr = lf.fit_poly(image.shape, leftx, lefty, ploty)
    else:
        left_fit_cr = process.left_crs[-1]
        left_fitx = left_fit_cr[0] * ploty ** 2 + left_fit_cr[1] * ploty + left_fit_cr[2]

    if len(rightx) > 0:
        right_fitx, right_fit_cr = lf.fit_poly(image.shape, rightx, righty, ploty)
    else:
        right_fit_cr = process.right_crs[-1]
        right_fitx = right_fit_cr[0] * ploty ** 2 + right_fit_cr[1] * ploty + right_fit_cr[2]

    left_curvature, right_curvature = cal.measure_curvature_real(left_fitx, right_fitx, ploty, params.xm, params.ym)
    position = cal.calulate_position(left_fitx, right_fitx, params.xm)

    lane_width, lane_width_min, lane_width_max = cal.calulate_lane_size(left_fitx, right_fitx, params.xm)
    current_sane = sanitizer.check_current_lanes(left_fit_cr, right_fit_cr, lane_width_max, lane_width_min, lane_width)

    prev_left_cr = left_fit_cr
    prev_right_cr = right_fit_cr

    if len(process.right_crs) > 0:
        prev_left_cr = process.left_crs[-1]
        prev_right_cr = process.right_crs[-1]

    if not current_sane:
        left_fit_cr = prev_left_cr
        right_fit_cr = prev_right_cr
        left_fitx = left_fit_cr[0] * ploty ** 2 + left_fit_cr[1] * ploty + left_fit_cr[2]
        right_fitx = right_fit_cr[0] * ploty ** 2 + right_fit_cr[1] * ploty + right_fit_cr[2]

    process.frame += 1

    process.left_crs.append(left_fit_cr)
    process.right_crs.append(right_fit_cr)
    process.lane_width_px.append(right_fitx[-1] - left_fitx[-1])
    process.sanity.append(current_sane)

    image_lines = renderer.draw_lanes(out_img,
                                      left_fitx,
                                      right_fitx,
                                      ploty)

    lanes_unwarped = lf.fromBirdsEye(image_lines, params.warp_x1, params.warp_horizon)
    result = renderer.weighted_img(lanes_unwarped, image)

    curvature = np.average([left_curvature, right_curvature])
    process.curvature.append(curvature)
    curvature_display = np.average(process.curvature[-10:])

    position_direction = "LEFT" if position > 0 else "RIGHT"

    position_text = "{:.2f}".format(np.abs(position))+" meters to the " + position_direction

    renderer.addText(result, ["Curvature: " + str(int(curvature_display)) + " m ",
                              "Vehicle position: " + position_text,
                              ])
    return result, out_img
