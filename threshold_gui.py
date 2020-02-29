import cv2
from src import parameters
from src import lane_finder
from src import camera
import argparse

class ThresholdGui:
    def __init__(self, image, params):
        self.image = image
        self._params = params
        self._smin = params.thresh_s[0]
        self._smax = params.thresh_s[1]
        self._sxmin = params.thresh_sx[0]
        self._sxmax = params.thresh_sx[1]

        def onSMinChanged(pos):
            self._smin = pos
            self._render()

        def onSMaxChanged(pos):
            self._smax = pos
            self._render()

        def onSxMinChanged(pos):
            self._sxmin = pos
            self._render()

        def onSxMaxChanged(pos):
            self._sxmax = pos
            self._render()


        cv2.namedWindow('result')
        cv2.namedWindow('warped')

        cv2.createTrackbar('s-min', 'result', self._smin, 255, onSMinChanged)
        cv2.createTrackbar('s-max', 'result', self._smax, 255, onSMaxChanged)
        cv2.createTrackbar('sx-min', 'result', self._sxmin, 255, onSxMinChanged)
        cv2.createTrackbar('sx-max', 'result', self._sxmax, 255, onSxMaxChanged)
        self._render()
        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('result')
        cv2.destroyWindow('warped')


    def _render(self):
        result = lane_finder.threshold(self.image, [self._smin, self._smax], [self._sxmin, self._sxmax]) * 255
        warped  = lane_finder.toBirdsEye(result, self._params.warp_x1, self._params.warp_x2, self._params.warp_horizon)
        cv2.imshow('result', result)
        cv2.imshow('warped', warped)


parser = argparse.ArgumentParser(description='Visualizes the threshold process.')
parser.add_argument('filename')

args = parser.parse_args()
image = cv2.imread(args.filename)

params = parameters.LaneFinderParams()
thresh = ThresholdGui(image, params)
