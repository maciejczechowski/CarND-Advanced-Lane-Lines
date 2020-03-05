import numpy as np
import cv2
from src import lane_finder as lf
from src import parameters
import argparse

class WarpFinder:
    def __init__(self, image, horizon = 400, x1 = 500):
        self.image1 = image
        self._horizon = horizon
        self._x1 = x1


        def onChangeHorizon(pos):
            self._horizon = pos
            self._render()

        def onChangeX1(pos):
            self._x1 = pos
            self._render()

        cv2.namedWindow('result')

        cv2.createTrackbar('horizon', 'result', self._horizon, 720, onChangeHorizon)
        cv2.createTrackbar('x1', 'result', self._x1, 640, onChangeX1)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('result')

    def draw_grid(self, img, w, h, line_color=(0, 255, 0), thickness=1, type_= cv2.LINE_AA, pxstep=50):

        '''(ndarray, 3-tuple, int, int) -> void
        draw gridlines on img
        line_color:
            BGR representation of colour
        thickness:
            line thickness
        type:
            8, 4 or cv2.LINE_AA
        pxstep:
            grid line frequency in pixels
        '''
        x = pxstep
        y = pxstep

        while x < w:
            cv2.line(img, (x, 0), (x, h), color=line_color, lineType=type_, thickness=thickness)
            x += pxstep

        while y < h:
            cv2.line(img, (0, y), (w, y), color=line_color, lineType=type_, thickness=thickness)
            y += pxstep


    def _render(self):
        warped1 = lf.toBirdsEye(self.image1, self._x1, self._horizon)

        self.draw_grid(warped1, 1280, 720)
        self._result = warped1

        cv2.imshow('result', self._result)

parser = argparse.ArgumentParser(description='Visualizes the warp transform.')
parser.add_argument('filename')

args = parser.parse_args()


image = cv2.imread(args.filename)

params = parameters.LaneFinderParams()
thresh = WarpFinder(image, params.warp_horizon, params.warp_x1)
