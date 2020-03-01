import numpy as np


# fits poly using given modifier (to px-to-m conversions)
def radius_of_curvature(pixels_x, pixels_y, mx, my):
    y_eval = np.max(pixels_y) * my
    fit = np.polyfit(pixels_y * my, pixels_x * mx, 2)
    curvature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
    return curvature


# calcualte curvature in meters, based on parabola
def measure_curvature_real(left_fit, right_fit, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension #todo: params
    xm_per_pix = 3.7 / 830

    left_curvature_real = radius_of_curvature(left_fit, ploty, xm_per_pix, ym_per_pix)
    right_curvature_real = radius_of_curvature(right_fit, ploty, xm_per_pix, ym_per_pix)

    return left_curvature_real, right_curvature_real


def calulate_position(left_fit, right_fit):
    xm_per_pix = 3.7 / 830  # meters per pixel in x dimension #todo: params

    left_pos = left_fit[-1]
    right_pos = right_fit[-1]

    center_px = left_pos + (right_pos - left_pos) / 2
    diff = center_px - 640
    diff_in_m = diff * xm_per_pix
    return diff_in_m


def calulate_lane_size(left_fit, right_fit):
    xm_per_pix = 3.7 / 830  # meters per pixel in x dimension

    lane_width_m = (right_fit - left_fit) * xm_per_pix  # todo: params

    lane_width = lane_width_m[-1]
    lane_width_min = np.min(lane_width_m)
    lane_width_max = np.max(lane_width_m)

    lane_diff = lane_width_m - lane_width
    residual = np.std(np.abs(lane_width_m - lane_width))

    diff = lane_width_max - lane_width_min
    return lane_width, lane_width_min, lane_width_max, residual


def check_lanes(left_fit_cr, right_fit_cr, ploty):
    # if A parameter changes sign it means that lanes ar goiung off
    if left_fit_cr[0] * right_fit_cr[0] < 0:
        return False

    return True
