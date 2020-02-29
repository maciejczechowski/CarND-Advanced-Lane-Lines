import numpy as np


class LaneFinderParams:
    # TODO: prepare beforehand
    camera_mtx = np.array(
        [[1.15777930e+03, 0.00000000e+00, 6.67111054e+02], [0.00000000e+00, 1.15282291e+03, 3.86128938e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    camera_dist = np.array([[-0.24688775, -0.02373132, -0.00109842, 0.00035108, -0.00258571]])

    warp_x1 = 561  # 563  # x position of upper-left point that will be transformed
    warp_x2 = 0
    warp_horizon = 452  # 460 # horizion position (y position of upper points that will be transformed
#around ~21m in roi

    thresh_s = (200, 254)  # (92, 254)  # threshold for S channel
    thresh_sx = (20, 244)  # (20, 244)  # threshold for L channel derivative

    lane_nwindows = 10  # number of windows to use in lane-finding
    lane_margin = 150  # margin around window center
    lane_minpix = 50  # number of pixels needed to recenter window


class LaneFinderProcess:
    frame = 0
