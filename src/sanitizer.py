import numpy as np
from . import calculations


# fits poly using given modifier (to px-to-m conversions)


def check_current_lanes(left_fit_cr,
                        right_fit_cr,
                        lane_size_min,
                        lane_size_max,
                        lane_size_calculated):

    if left_fit_cr is None or right_fit_cr is None:
        return False

    # b coefficient should not differ by more than 0.7
    if np.abs(left_fit_cr[1] - right_fit_cr[1]) > 0.5:
        return False

    # lane size should be around ~4.7
#    if lane_size_calculated < 3.5 or lane_size_calculated > 4.2:
    if lane_size_calculated > 4.2:

        return False

    # lanes should be mostly parallel (lane width should not differ by more than ~1m)
    if np.abs(lane_size_max - lane_size_min > 1):
        return False

    return True


def check_with_previous(current_left_fit_cr,
                        current_right_fit_cr,
                        previous_left_fit_cr,
                        previous_right_fit_cr):

    if current_left_fit_cr is None or current_right_fit_cr is None or previous_left_fit_cr is None or previous_right_fit_cr is None:
        return False, False

    left_sane = True
    right_sane = True

    # b coefficient should not change more than 0.5
    if np.abs(current_left_fit_cr[1] - previous_left_fit_cr[1] > 0.5):
        left_sane = False

    if np.abs(current_right_fit_cr[1] - previous_right_fit_cr[1] > 0.5):
        right_sane = False

    # c coefficent should not change more than 20
    if np.abs(current_left_fit_cr[2] - previous_left_fit_cr[2] > 20):
        left_sane = False

    if np.abs(current_right_fit_cr[2] - previous_right_fit_cr[2] > 20):
        right_sane = False

    return left_sane, right_sane
