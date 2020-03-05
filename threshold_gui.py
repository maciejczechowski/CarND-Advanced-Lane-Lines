import cv2
from src import parameters, renderer
from src import lane_finder
from src import camera
import argparse
import matplotlib.image as mpimg
import numpy as np

class ThresholdGui:
    def __init__(self, image, params):
        self.image = image
        self.params = params

        def onSMinChanged(pos):
            self.params.thresh_s = (pos, self.params.thresh_s[1])
            self._render()

        def onSMaxChanged(pos):
            self.params.thresh_s = (self.params.thresh_s[0], pos)
            self._render()

        def onSxMinChanged(pos):
            self.params.thresh_sx = (pos, self.params.thresh_sx[1])
            self._render()

        def onSxMaxChanged(pos):
            self.params.thresh_sx = (self.params.thresh_sx[0], pos)
            self._render()

        def onHMinChanged(pos):
            self.params.thresh_h = (pos, self.params.thresh_h[1])
            self._render()

        def onHMaxChanged(pos):
            self.params.thresh_h = (self.params.thresh_h[0], pos)
            self._render()

        cv2.namedWindow('result')
        cv2.namedWindow('warped')
        cv2.namedWindow('params')

        cv2.createTrackbar('s-min', 'params', self.params.thresh_s[0], 255, onSMinChanged)
        cv2.createTrackbar('s-max', 'params', self.params.thresh_s[1], 255, onSMaxChanged)
        cv2.createTrackbar('sx-min', 'params', self.params.thresh_sx[0], 255, onSxMinChanged)
        cv2.createTrackbar('sx-max', 'params', self.params.thresh_sx[1], 255, onSxMaxChanged)
        cv2.createTrackbar('h-min', 'params', self.params.thresh_h[0], 255, onHMinChanged)
        cv2.createTrackbar('h-max', 'params', self.params.thresh_h[1], 255, onHMaxChanged)
        self._render()
        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('result')
        cv2.destroyWindow('warped')


    def _render(self):
        result = lane_finder.threshold(self.image,
                                       self.params.thresh_s,
                                       self.params.thresh_sx,
                                       self.params.thresh_h) * 255

        warped = lane_finder.toBirdsEye(result, self.params.warp_x1, self.params.warp_x2, self.params.warp_horizon)

        res2 = np.dstack((result, result, result))
        r2 = renderer.weighted_img(res2, self.image)
        r2 = cv2.cvtColor(r2, cv2.COLOR_RGB2BGR)
        cv2.imshow('result', r2)
        cv2.imshow('warped', warped)


parser = argparse.ArgumentParser(description='Visualizes the threshold process.')
parser.add_argument('filename')

args = parser.parse_args()
image = mpimg.imread(args.filename)

params = parameters.LaneFinderParams()
thresh = ThresholdGui(image, params)
