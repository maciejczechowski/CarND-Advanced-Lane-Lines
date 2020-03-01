import numpy as np
import cv2
from src import camera
from src import parameters
from src import pipeline
import argparse
import matplotlib.image as mpimg

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
        cv2.namedWindow('lanes')

        cv2.createTrackbar('#windows', 'result', self._params.lane_nwindows, 20, onChangeNWindows)
        cv2.createTrackbar('margin', 'result', self._params.lane_margin, 300, onChangeMargin)
        cv2.createTrackbar('minpix', 'result', self._params.lane_minpix, 300, onChangeMinpix)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('result')
        cv2.destroyWindow('lanes')


    def _render(self):
        result, lanes = pipeline.process_image(self.image1,  self._params)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        lanes = cv2.cvtColor(lanes, cv2.COLOR_RGB2BGR)
        cv2.imshow('result', result)
        cv2.imshow('lanes', lanes)

parser = argparse.ArgumentParser(description='Visualizes the histogram based lane finder.')
parser.add_argument('filename')

args = parser.parse_args()


image = mpimg.imread(args.filename)

params = parameters.LaneFinderParams()
thresh = WarpFinder(image, params)
