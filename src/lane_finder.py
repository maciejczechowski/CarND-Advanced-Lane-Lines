import numpy as np
import cv2
from . import camera
import matplotlib.pyplot as plt

def get_points(x1, x2, horizon):
    srcpoints = np.array([[x1, horizon],
                          [1280 - x1, horizon],
                          [300, 700],
                          [980, 700]
                          ], dtype="float32")

    dstpoints = np.array([[310, 0],
                          [970, 0],
                          [310, 720],
                          [970, 720]
                          ], dtype="float32")

    return srcpoints, dstpoints

def toBirdsEye(img, x1, x2, horizon):
    srcpoints, dstpoints = get_points(x1, x2, horizon)
    warped = camera.warp(img, srcpoints, dstpoints)
    return warped


def fromBirdsEye(img, x1, x2, horizon):
    srcpoints, dstpoints = get_points(x1, x2, horizon)
    warped = camera.warp(img, dstpoints, srcpoints)
    return warped

def color_prepare(image):
    denoised = cv2.fastNlMeansDenoisingColored(image)
    return denoised

# threshold image based on S channel and Sobel function on L channel
def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100), h_thresh=(20, 30)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_yellow = cv2.inRange(hsv, (h_thresh[0], 60, 100), (h_thresh[1], 255, 255))
    mask_yellow_binary = np.zeros_like(mask_yellow)
    mask_yellow_binary[(mask_yellow == 255)] = 1

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelxy = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1, ksize=11)  # Take the derivative in x, y

    abs_sobelx = np.abs(sobelxy) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(l_channel >= s_thresh[0]) & (l_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = sxbinary | s_binary | mask_yellow_binary
    return color_binary


def find_start_windows_histogram(binary_warped, nwindows):
    window_height = np.int(binary_warped.shape[0] // nwindows)
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    lefty_base = binary_warped.shape[0] - window_height
    righty_base = binary_warped.shape[0] - window_height

    return leftx_base, lefty_base, rightx_base, righty_base, window_height

def find_start_windows(binary_warped, nwindows=9, activation=20):
    window_height = np.int(binary_warped.shape[0] // nwindows)
    height = binary_warped.shape[0]

    # Take a histogram of the bottom half

    leftx_base = -1
    lefty_base = -1
    rightx_base = -1
    righty_base = -1

    for vwindow in range(height - window_height, 0, -window_height):
        histogram = np.sum(binary_warped[vwindow:vwindow+window_height, :], axis=0)

        midpoint = np.int(histogram.shape[0] // 2)
        lmax = np.argmax(histogram[:midpoint])
        rmax = np.argmax(histogram[midpoint:]) + midpoint

        if leftx_base == -1 and histogram[lmax] >= activation:
            lefty_base = vwindow
            leftx_base = lmax

        if rightx_base == -1 and histogram[rmax] >= activation:
            righty_base = vwindow
            rightx_base = rmax

        if rightx_base > -1 and leftx_base > -1:
            break

    return leftx_base, lefty_base, rightx_base, righty_base, window_height

def window_trace(binary_img,
                 ref_img,
                 window_center,
                 window_top,
                 window_height,
                 margin,
                 minpix,
                 maxempty=4):

    out_img = ref_img

    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])  # all non-zero y
    nonzerox = np.array(nonzero[1])  # all non-zero x

    win_y_low = window_top
    win_y_high = window_top + window_height

    empty_windows = 0

    lane_inds = []
    num_found = 0
    while win_y_low > 0:
        # Identify window boundaries in x and y (and right and left)
        win_x_low = window_center - margin
        win_x_high = window_center + margin

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                nonzerox < win_x_high)).nonzero()[0]

        if len(good_inds) < minpix:
            empty_windows += 1
        else:
            num_found += 1
            lane_inds.append(good_inds)

            lx = nonzerox[good_inds]
            ly = nonzeroy[good_inds]

            cv2.rectangle(out_img,
                          (win_x_low, win_y_low),
                          (win_x_high, win_y_high),
                          (0, 255, 0),
                          2)
            out_img[ly, lx] = (0, 0, 255)
        if empty_windows >= maxempty:
            break

        if len(good_inds) > minpix:
            window_center = np.int(np.mean(nonzerox[good_inds]))

        win_y_low = win_y_low - window_height
        win_y_high = win_y_high - window_height

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        lane_inds = np.concatenate(lane_inds)
    except ValueError:
        pass

    # Extract left and right line pixel positions
    lanex = nonzerox[lane_inds]
    laney = nonzeroy[lane_inds]

    return lanex, laney, num_found, out_img


# finds a lanes based on histogram and moving windows
# parameters:
#  binary_warped - 1d, birds eye view image to analyze
#  nwindows - number of windows to use
#  margin - margin to use around windows center (defines window width)
#  minpix - minimum number of 1-valued pixels in window in order to recenter window
def find_lane_pixels_v2(binary_img,
                        nwindows=9,
                        margin=100,
                        minpix=30,
                        maxemptywindows=2):

    ref_img = np.dstack((binary_img, binary_img, binary_img)) * 255
    leftx_base, lefty_base, rightx_base, righty_base, window_height = find_start_windows_histogram(binary_img, nwindows) #find_start_windows(binary_img, nwindows, 15)
    leftx, lefty, lfound, lout_img = window_trace(binary_img, ref_img, leftx_base, lefty_base, window_height, margin, minpix)
    rightx, righty, rfound, rout_img = window_trace(binary_img, ref_img, rightx_base, righty_base, window_height, margin, minpix)


    return leftx, lefty, lfound, rightx, righty, rfound, ref_img


# fits the parabola f(y) based on points provied
def fit_poly(img_shape, x, y, ploty):
    fit = np.polyfit(y, x, 2)

    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

    return fitx, fit

