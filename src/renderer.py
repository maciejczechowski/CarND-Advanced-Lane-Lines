import numpy as np
import cv2
from . import camera
# Functions used to produce final images based on calculated data

# Returns an image with lanes drawn using the points specified
def draw_lanes(image,
               left_fitx,
               right_fitx,
               ploty,
               lane_color=(100, 200, 100),
               thickness=15
               ):
    out_img = np.copy(image)
    pts_left = np.column_stack([left_fitx, ploty])
    pts_right = np.column_stack([right_fitx, ploty])
    cv2.polylines(out_img, np.int32([pts_left, pts_right]), False, color=lane_color, thickness=thickness)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(out_img, np.int32([pts]), color=lane_color)

    return out_img

"""
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_left_posed =  camera.warp(pts_left, dst, src)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts_right_posed = camera.warp(pts_right, dst, src)
    pts = np.hstack((pts_left_posed, pts_right_posed))

    cv2.fillPoly(out_img, np.int32([pts]), color=lane_color)
"""



#blends two images
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)