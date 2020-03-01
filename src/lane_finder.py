import numpy as np
import cv2
from . import camera
import matplotlib.pyplot as plt

def toBirdsEye(img, x1, x2, horizon):
    srcpoints = np.array([[x1, horizon],
                          [1280 - x1, horizon],
                          [x2, 700],
                          [1280 - x2, 700]
                          ], dtype="float32")

    dstpoints = np.array([[0, 0],
                          [1280, 0],
                          [0, 720],
                          [1280, 720]
                          ], dtype="float32")
    warped = camera.warp(img, srcpoints, dstpoints)
    return warped


def fromBirdsEye(img, x1, x2, horizon):
    srcpoints = np.array([[x1, horizon],
                          [1280 - x1, horizon],
                          [x2, 700],
                          [1280 - x2, 700]
                          ], dtype="float32")

    dstpoints = np.array([[0, 0],
                          [1280, 0],
                          [0, 720],
                          [1280, 720]
                          ], dtype="float32")
    warped = camera.warp(img, dstpoints, srcpoints)
    return warped


def threshold_mask(img, thresh=128, maxval=255, type=cv2.THRESH_BINARY):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshed = cv2.threshold(img, thresh, maxval, type)[1]
    return threshed

def smooth_mask(mask, kernel_size=11):
    blurred  = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    threshed = threshold_mask(blurred)
    return threshed

def dilate_mask(mask, kernel_size=15):
    kernel  = np.ones((kernel_size,kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


#discards the dark area of image so dark spots on the road are not identified as lanes
def color_prepare(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    maskWhite = cv2.inRange(hsv, (0,0,230), (255, 255, 255))
    maskYellow = cv2.inRange(hsv, (20,20,100), (30, 255, 255))
    mask = cv2.bitwise_or(maskWhite, maskYellow)

    mask = dilate_mask(mask)
    mask = smooth_mask(mask, 11)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    target = cv2.convertScaleAbs(blurred*(1-alpha) + image * alpha)

    return target

# threshold image based on S channel and Sobel function on L channel
def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100), h_thresh=(20, 30)):
    img = np.copy(img)

    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clr = clahe.apply(r_channel)
    clg = clahe.apply(g_channel)
    clb = clahe.apply(b_channel)

    img = np.dstack([clr, clg, clb])

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))


    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    sxbinary[(l_channel >= sx_thresh[0]) & (l_channel <= sx_thresh[1])] = 1

    # inlcude yellow

  #  mask_yellow = cv2.inRange(hls, (h_thresh[0], 0, 100), (h_thresh[1], 255, 255))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_yellow = cv2.inRange(hsv, (h_thresh[0], 20, 100), (h_thresh[1], 255, 255))
    mask_yellow_binary = np.zeros_like(mask_yellow)
    mask_yellow_binary[(mask_yellow == 255)] = 1
  #  mask_yellow_binary = np.zeros_like(h_channel)
  #  mask_yellow_binary[(h_channel >= h_thresh[0])
   #                    & (h_channel < h_thresh[1])
    #                   & (l_channel >= s_thresh[0])
     #                  & (l_channel < s_thresh[1])] = 1

    # Threshold x gradient
    # TODO: lepiej percentile?

    # Threshold color channel
    highestintensity = s_thresh[0]  # np.percentile(s_channel, s_thresh[0])
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= highestintensity) & (s_channel <= s_thresh[1])] = 1


    # Stack each channel
    color_binary = sxbinary | s_binary | mask_yellow_binary
    return color_binary


# finds a lanes based on histogram and moving windows
# parameters:
#  binary_warped - 1d, birds eye view image to analyze
#  nwindows - number of windows to use
#  margin - margin to use around windows center (defines window width)
#  minpix - minimum number of 1-valued pixels in window in order to recenter window
def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50, maxemptywindows=2):
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Take a histogram of the bottom half
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])  # tablica wszystkich y niezerowych
    nonzerox = np.array(nonzero[1])  # tablica wszystkich x niezerowych
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    empty_left = 0
    empty_right = 0
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

        if len(good_right_inds) == 0:
            empty_right += 1
        elif empty_right < maxemptywindows:
            right_lane_inds.append(good_right_inds)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        if len(good_left_inds) == 0:
            empty_left += 1
        elif empty_left < maxemptywindows:
            left_lane_inds.append(good_left_inds)
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


# fits the parabola f(y) based on points provied
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit
