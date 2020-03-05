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

    # b coefficient should not differ by more than 0.5
    if np.abs(left_fit_cr[1] - right_fit_cr[1]) > 0.5:
        return False

    # lane size should be around ~3.7 so ignore too wide ones
    if lane_size_calculated > 4.2:
        return False

    # lanes should be mostly parallel (lane width should not differ by more than ~1m)
    if np.abs(lane_size_max - lane_size_min > 1):
        return False

    return True
