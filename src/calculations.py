import numpy as np

def radius_of_curvature(pixels_x, pixels_y, mx, my):
    if pixels_y is None or pixels_x is None:
        return 0

    y_eval = np.max(pixels_y) * my
    fit = np.polyfit(pixels_y * my, pixels_x * mx, 2)
    curvature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
    return curvature


# calcualte curvature in meters, based on parabola
def measure_curvature_real(left_fit, right_fit, ploty, xm, ym):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    left_curvature_real = radius_of_curvature(left_fit, ploty, xm, ym)
    right_curvature_real = radius_of_curvature(right_fit, ploty, xm, ym)

    return left_curvature_real, right_curvature_real


def calulate_position(left_fit, right_fit, xm):
    if left_fit is None or right_fit is None:
        return 0
    left_pos = left_fit[-1]
    right_pos = right_fit[-1]

    center_px = left_pos + (right_pos - left_pos) / 2
    diff = center_px - 640
    diff_in_m = diff * xm
    return diff_in_m


def calulate_lane_size(left_fit, right_fit, xm):
    if left_fit is None or right_fit is None:
        return 0, 0, 0, 0

    lane_width_m = (right_fit - left_fit) * xm

    lane_width = lane_width_m[-1]
    lane_width_min = np.min(lane_width_m)
    lane_width_max = np.max(lane_width_m)

    return lane_width, lane_width_min, lane_width_max
