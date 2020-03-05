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


def extrapolate_cr(history_cr, ploty):
    num = len(history_cr)
    x = np.arange(0, num)
    y = np.array(history_cr)
    fa = interpolate.interp1d(x, y[::, 0], fill_value="extrapolate")
    fb = interpolate.interp1d(x, y[::, 1], fill_value="extrapolate")
    fc = interpolate.interp1d(x, y[::, 2], fill_value="extrapolate")
    fit_cr = [fa(num), fb(num), fc(num)]
    fit_x = fit_cr[0] * ploty ** 2 + fit_cr[1] * ploty + fit_cr[2]
    return fit_cr, fit_x

def limit_change(current_cr, prev_cr, ploty):
    b_diff = np.clip(current_cr[1] - prev_cr[1], -0.5, 0.5)
    c_diff = np.clip(current_cr[2] - prev_cr[2], -30, 30)
    a = current_cr[0]
    b = prev_cr[1] + b_diff
    c = prev_cr[2] + c_diff

    fit_cr = [a, b, c]
    fitx = fit_cr[0] * ploty ** 2 + fit_cr[1] * ploty + fit_cr[2]

    return fit_cr, fitx


def extrapolate_frame(left_ok, right_ok, left_fit_cr, right_fit_cr, ploty, process):
    message = ""
    if not left_ok:
        left_to_extrapolate = process.left_crs[-10:]
        left_fit_cr, left_fitx = extrapolate_cr(left_to_extrapolate, ploty)
        message = "EXTRAPOLATE"
    else:
        left_fitx = left_fit_cr[0] * ploty ** 2 + left_fit_cr[1] * ploty + left_fit_cr[2]

    if not right_ok:
        right_to_extrapolate = process.right_crs[-10:]
        right_fit_cr, right_fitx = extrapolate_cr(right_to_extrapolate, ploty)
        message = "EXTRAPOLATE"
    else:
        right_fitx = right_fit_cr[0] * ploty ** 2 + right_fit_cr[1] * ploty + right_fit_cr[2]

    return left_fit_cr, left_fitx, right_fit_cr, right_fitx, message


def process_image(image, params, process: LaneFinderProcess):
    image = camera.undistort(image, params.camera_mtx, params.camera_dist)
    image_thresholded = lf.threshold(image, params.thresh_s, params.thresh_sx, params.thresh_h)
    image_warped = lf.toBirdsEye(image_thresholded, params.warp_x1, params.warp_x2, params.warp_horizon)

    leftx, lefty, lfound, rightx, righty, rfound, out_img = lf.find_lane_pixels_v2(image_warped,
                                                                                   params.lane_nwindows,
                                                                                   params.lane_margin,
                                                                                   params.lane_minpix)

    left_fitx = None
    right_fitx = None
    left_fit_cr = None
    right_fit_cr = None

    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

    if (len(leftx) == 0 or len(rightx) == 0) and len(process.right_crs) < 2: # fatal, skip frame
        return image, out_img

    if len(leftx) > 0:
        left_fitx, left_fit_cr = lf.fit_poly(image.shape, leftx, lefty, ploty)
    else:
        left_fit_cr, left_fitx = extrapolate_cr(process.left_crs[-10:], ploty)

    if len(rightx) > 0:
        right_fitx, right_fit_cr = lf.fit_poly(image.shape, rightx, righty, ploty)
    else:
        left_fit_cr, left_fitx = extrapolate_cr(process.left_crs[-10:], ploty)

    left_curvature, right_curvature = cal.measure_curvature_real(left_fitx, right_fitx, ploty, params.xm, params.ym)
    position = cal.calulate_position(left_fitx, right_fitx, params.xm)
    lane_width, lane_width_min, lane_width_max, avg_diff = cal.calulate_lane_size(left_fitx, right_fitx, params.xm)

    current_sane = sanitizer.check_current_lanes(left_fit_cr, right_fit_cr, lane_width_max, lane_width_min, lane_width)


    left_ok = False
    right_ok = False
    prev_left_cr = left_fit_cr
    prev_right_cr = right_fit_cr

    if len(process.right_crs) > 0:
        prev_left_cr = process.left_crs[-1]
        prev_right_cr = process.right_crs[-1]
        left_ok, right_ok = sanitizer.check_with_previous(left_fit_cr, right_fit_cr, prev_left_cr,
                                                          prev_right_cr)
    # sanitize #found

    message = ""
    if not current_sane:
        left_fit_cr = prev_left_cr
        right_fit_cr = prev_right_cr
        left_fitx = left_fit_cr[0] * ploty ** 2 + left_fit_cr[1] * ploty + left_fit_cr[2]
        right_fitx = right_fit_cr[0] * ploty ** 2 + right_fit_cr[1] * ploty + right_fit_cr[2]
        message = "EXTRA_FAIL_1"
      #  left_fit_cr, left_fitx, right_fit_cr, right_fitx, message =

        #extrapolate_frame(left_ok, right_ok, left_fit_cr, right_fit_cr, ploty, process)

    lane_width, lane_width_min, lane_width_max, avg_diff = cal.calulate_lane_size(left_fitx, right_fitx, params.xm)
    sane_after_interpolate = sanitizer.check_current_lanes(left_fit_cr, right_fit_cr, lane_width_min, lane_width_max, lane_width)
    if not sane_after_interpolate: # we screwed. just use previous frame data
        left_fit_cr = prev_left_cr
        right_fit_cr = prev_right_cr
        left_fitx = left_fit_cr[0] * ploty ** 2 + left_fit_cr[1] * ploty + left_fit_cr[2]
        right_fitx = right_fit_cr[0] * ploty ** 2 + right_fit_cr[1] * ploty + right_fit_cr[2]
        message = "EXTRA_FAIL"


    process.frame += 1

    process.left_crs.append(left_fit_cr)
    process.right_crs.append(right_fit_cr)
    process.lane_width_px.append(right_fitx[-1] - left_fitx[-1])
    process.sanity.append(current_sane)

    image_lines = renderer.draw_lanes(out_img,
                                      left_fitx,
                                      right_fitx,
                                      ploty)

    lanes_unwarped = lf.fromBirdsEye(image_lines, params.warp_x1, params.warp_x2, params.warp_horizon)
    result = renderer.weighted_img(lanes_unwarped, image)

    image_warped = lf.toBirdsEye(image, params.warp_x1, params.warp_x2, params.warp_horizon)

  #  result = renderer.weighted_img(image_lines, image_warped)
    # sanity = sanitizer.check_lanes(left_fit_cr, right_fit_cr, ploty)

    renderer.addText(result, ["LC: " + str(left_curvature),
                              "RC: " + str(right_curvature),
                              "position: " + str(position),
                              "lane_width: " + str(lane_width),
                              "lane max " + str(lane_width_max),
                              "lane min" + str(lane_width_min),
                              # "aD" + str(np.abs(left_fit_cr[0] - right_fit_cr[0])),
                              # "bD" + str(np.abs(left_fit_cr[1] - right_fit_cr[1])),
                              "sanity " + str(current_sane),
                              "left_temporal " + str(left_ok),
                              "right_temporal " + str(right_ok),
                              "lad " + message,
                              ])
    #  wimg = np.dstack((image_warped * 255, image_warped, image_warped))
    # r1 = renderer.weighted_img(out_img, wimg)
    return result, out_img
