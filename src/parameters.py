import numpy as np


class LaneFinderParams:
    camera_mtx = np.array(
        [[1.15777930e+03, 0.00000000e+00, 6.67111054e+02], [0.00000000e+00, 1.15282291e+03, 3.86128938e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    camera_dist = np.array([[-0.24688775, -0.02373132, -0.00109842, 0.00035108, -0.00258571]])

    # move2  x1: 506, h = 498

#    warp_x1 = 589  # 563  # x position of upper-left point that will be transformed
#    warp_x2 = 0
#    warp_horizon = 460

    warp_x1 = 597  # 563  # x position of upper-left point that will be transformed
    warp_x2 = 0
    warp_horizon = 472

    # 460 # horizion position (y position of upper points that will be transformed
    # around ~21m in roi

    thresh_s = (170, 255)  # (92, 254)  # threshold for S channel
    thresh_h = (15, 30)  # threshold for H channel
    #  thresh_sx = (20, 244)  # (20, 244)  # threshold for L channel derivative
    thresh_sx = (212, 255)  # (20, 244)  # threshold for L channel derivative

    lane_nwindows = 7  # number of windows to use in lane-finding
    lane_margin = 150  # margin around window center
    lane_minpix = 30  # number of pixels needed to recenter window

    xm = 3.7 / 815 # meters per pixel (x)
    ym = 30 / 720  # meters per pixel (y)

class LaneFinderProcess:
    frame = 0

    sanity = []  # array indicating which frames were calculated as sane


    lane_width_px = []

    left_crs = []  # array of calculated left_fir_cr paramaters from video frame
    right_crs = []  # array of calculated right_fir_cr paramaters from video frame
