import tkMessageBox
from Tkinter import *

import cv2
import numpy as np
import time


class LaserBoard:
    def __init__(self, BoardHeight, BoardWidth):
        Tk().withdraw()
        self.calibratedBool = False
        self.cornerCoordinates = np.zeros([4, 2])
        self.cornerCount = 0

        self.dotFinder = Cat()
        self.vid = cv2.VideoCapture(0)
        self.BoardH = BoardHeight
        self.BoardW = BoardWidth
        self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
        self.dotsFinder = Cat()
        self.H = []

    def start(self):
        #try:
            self.calibrationSetup()
            self.laserBoardRun()
        #except Exception, e:
        #    print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        #    print str(e)
            self.release()

    def laserBoardRun(self):

        while (1):
            ret, view = self.vid.read()

            if cv2.waitKey(5) & 0xFF == ord('q'):
                lb.release()
            elif cv2.waitKey(5) & 0xFF == ord('r'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            view, dots, status = self.dotsFinder.find(board)
            self.canvas = np.maximum(self.canvas, view)

            catView = cv2.drawKeypoints(board,dots,np.array([]),(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if status[3] == 2:
                print 'doubleClicked'
            elif status[3] == 1 and status[2]:
                print 'clicked'

            cv2.imshow('setup', catView)
            cv2.imshow('Board', 255 - self.canvas)

    def calibrationSetup(self):
        cv2.namedWindow('Board')
        cv2.imshow('Board', 255 - self.canvas)
        cv2.namedWindow('setup')
        self.positionSetup()
        self.colorSetup()

    def cornersClicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if not self.calibratedBool:
                self.cornerCoordinates[self.cornerCount, :] = [x, y]
                self.cornerCount += 1
                if self.cornerCount == 4:
                    self.calibratedBool = True

    def positionSetup(self):

        cv2.setMouseCallback('setup', self.cornersClicked)
        ret, view = self.vid.read()
        tkMessageBox.showinfo("Calibration Instructions",
                              "Click the top left corner of the board and click the other corners in a clockwise fashion")
        while (1):
            if cv2.waitKey(5) & 0xFF == ord('e'):
                if self.calibratedBool:
                    self.H = cv2.getPerspectiveTransform(np.float32(self.cornerCoordinates), np.float32(
                        [[0, 0], [self.BoardW, 0], [self.BoardW, self.BoardH], [0, self.BoardH]]))
                    return
                else:
                    tkMessageBox.showinfo("Not Enough Corners selected", "Press R to reset Corners or q to exit")
            elif cv2.waitKey(5) & 0xFF == ord('r'):
                self.cornerCount = 0
                self.cornerCoordinates = np.zeros([4, 2])
                self.calibratedBool = False
            elif cv2.waitKey(5) & 0xFF == ord('q'):
                self.release()

            ret, view = self.vid.read()
            for i in self.cornerCoordinates:
                cv2.circle(view, tuple(int(x) for x in i), 5, (255, 0, 0), -1)
            cv2.imshow('setup', view)

    def colorSetup(self):
        def nothing(x):
            pass

            # Slider gui initialization

        cv2.namedWindow('control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('control', 300, 400)
        cv2.createTrackbar('R_min', 'control', 0, 360, nothing)
        cv2.createTrackbar('R_max', 'control', 0, 360, nothing)
        cv2.createTrackbar('G_min', 'control', 0, 360, nothing)
        cv2.createTrackbar('G_max', 'control', 0, 360, nothing)
        cv2.createTrackbar('B_min', 'control', 0, 360, nothing)
        cv2.createTrackbar('B_max', 'control', 0, 360, nothing)
        cv2.createTrackbar('manual/auto', 'control', 0, 1, nothing)
        cv2.createTrackbar('vid/img', 'control', 0, 1, nothing)

        cv2.setTrackbarPos('R_max', 'control', 360)
        cv2.setTrackbarPos('G_max', 'control', 360)
        cv2.setTrackbarPos('B_max', 'control', 360)

        while (True):
            if not cv2.getTrackbarPos('vid/img', 'control'):
                ret, frame = self.vid.read()
                frame = cv2.warpPerspective(frame, self.H, (self.BoardW, self.BoardH))
            else:
                frame = cv2.imread('Gamut.png', -1)

            if not cv2.getTrackbarPos('manual/auto', 'control'):
                r_min = cv2.getTrackbarPos('R_min', 'control')
                r_max = cv2.getTrackbarPos('R_max', 'control')
                g_min = cv2.getTrackbarPos('G_min', 'control')
                g_max = cv2.getTrackbarPos('G_max', 'control')
                b_min = cv2.getTrackbarPos('B_min', 'control')
                b_max = cv2.getTrackbarPos('B_max', 'control')
            else:
                r_min = int(max(frame[:,:,0].flatten()))
                r_max = 360
                g_min = int(max(frame[:,:,1].flatten()))
                g_max = 360
                b_min = int(max(frame[:,:,2].flatten()))
                b_max = 360
                cv2.setTrackbarPos('R_min', 'control', r_min)
                cv2.setTrackbarPos('B_min', 'control', b_min)
                cv2.setTrackbarPos('G_min', 'control', g_min)
                cv2.setTrackbarPos('vid/img', 'control', 0)

            CDThresh = cv2.inRange(frame, np.array([b_min, g_min, r_min]), np.array([b_max, g_max, r_max]))
            cv2.imshow('setup', cv2.bitwise_and(frame, frame, mask=CDThresh))

            if cv2.waitKey(5) & 0xFF == ord('e'):
                self.dotsFinder.trainEyes(np.array([b_min, g_min, r_min]), np.array([b_max, g_max, r_max]))
                cv2.destroyWindow('control')
                return
            elif cv2.waitKey(5) & 0xFF == ord('r'):
                cv2.setTrackbarPos('R_min', 'control', 0)
                cv2.setTrackbarPos('R_max', 'control', 180)
                cv2.setTrackbarPos('G_min', 'control', 0)
                cv2.setTrackbarPos('G_max', 'control', 360)
                cv2.setTrackbarPos('B_min', 'control', 0)
                cv2.setTrackbarPos('B_max', 'control', 360)
            elif cv2.waitKey(5) & 0xFF == ord('q'):
                self.release()

    def release(self):
        self.vid.release()
        cv2.destroyAllWindows()
        quit()


class Cat:
    def __init__(self):
        self.minBGR = [0,0,0]
        self.maxBGR = [360,360,360]
        self.time = 0
        self.numClicks = 0
        self.pressed = False

        #status indices:
        #0. previous time pressed/depressed
        #1. current time
        #2. pressed or depressed
        #3. number of times clicked [0/1/2]
        #4. double click delay threshold

        self.status = [0] * 5
        self.status[4] = .5

        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByArea = True
        params.minArea = 30

        self.blobDetector = cv2.SimpleBlobDetector_create(params)

    def trainEyes(self, minBGR, maxBGR):
        self.minBGR = minBGR
        self.maxBGR = maxBGR

    def find(self, view):
        dots = cv2.inRange(view, self.minBGR, self.maxBGR)
        dotsPos = self.blobDetector.detect(view)
        if len(dotsPos):
            if not self.status[2]:
                self.status[2] = True
                self.status[1] = time.time()
                if self.status[3] == 0:
                    self.status[3] == 1
                else:
                    if self.status[1] - self.status[0] < self.status[4]:
                        self.status[3] == 2
                    else:
                        self.status[3] == 1
                self.status[0] = self.status[1]
        else:
            if self.status[2]:
                self.status[2] = False
                self.status[1] = time.time()
                if self.status[3] == 2:
                    self.status[3] == 0
        return dots, dotsPos, self.status


if __name__ == "__main__":
    res = 200
    lb = LaserBoard(3 * res, 4 * res)
    lb.start()
