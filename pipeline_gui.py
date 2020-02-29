import numpy as np
import cv2
from src import camera
from src import parameters
from src import pipeline
import argparse

class WarpFinder:
    def __init__(self, image, params):
        self.image1 = image
        self._params = params



        def onChangeNWindows(pos):
            self._params.lane_nwindows = pos
            self._render()

        def onChangeMargin(pos):
            self._params.lane_margin = pos
            self._render()

        def onChangeMinpix(pos):
            self._params.lane_minpix = pos
            self._render()

        cv2.namedWindow('result')

        cv2.createTrackbar('#windows', 'result', self._params.lane_nwindows, 20, onChangeNWindows)
        cv2.createTrackbar('margin', 'result', self._params.lane_margin, 300, onChangeMargin)
        cv2.createTrackbar('minpix', 'result', self._params.lane_minpix, 300, onChangeMinpix)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('result')


    def _render(self):
        result = pipeline.process_image(self.image1,  self._params)
        cv2.imshow('result', result)

parser = argparse.ArgumentParser(description='Visualizes the histogram based lane finder.')
parser.add_argument('filename')

args = parser.parse_args()


image = cv2.imread(args.filename)

params = parameters.LaneFinderParams()
thresh = WarpFinder(image, params)
