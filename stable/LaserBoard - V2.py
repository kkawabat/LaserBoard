import cv2
import numpy as np
import time


class LaserBoard:
    def __init__(self, board_height, board_width):
        self.calibratedBool = False
        self.cornerCoordinates = np.zeros([4, 2])
        self.cornerCount = 0
        self.dotFinder = Cat()
        self.vid = cv2.VideoCapture(0)
        self.BoardH = board_height
        self.BoardW = board_width
        self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
        self.dotsFinder = Cat()
        self.H = []

    def start(self):
        # try:
            cv2.namedWindow('Board')
            cv2.namedWindow('setup')
            self.calibration_setup()
            self.laserBoardRun()
        # except Exception, e:
        #    print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        #    print str(e)
            self.release()

    def laserBoardRun(self):
        while 1:
            ret, view = self.vid.read()
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'):
                self.release()
            elif keypress == ord('r'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                print('canvas cleared')
            elif keypress == ord('c'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                self.calibration_setup()

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))

            dt_view, dots, status = self.dotsFinder.find(board, self.canvas)

            self.canvas = np.maximum(self.canvas, dt_view)

            cat_view = cv2.drawKeypoints(board, dots, np.array([]),
                                        (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if status[3] == 2:
                print 'doubleClicked'
            elif status[3] == 1 and status[2]:
                print 'clicked'

            cv2.imshow('setup', cat_view)
            cv2.imshow('Board', 255 - self.canvas)

    def calibration_setup(self):
        self.position_setup()
        self.color_setup()

    def corners_clicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if not self.calibratedBool:
                self.cornerCoordinates[self.cornerCount, :] = [x, y]
                self.cornerCount += 1
                if self.cornerCount == 4:
                    self.calibratedBool = True

    def position_setup(self):
        cv2.setMouseCallback('setup', self.corners_clicked)
        print 'calibrating screen corners please click corners from top left going clockwise'
        while 1:
            ret, view = self.vid.read()
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                if self.calibratedBool:
                    self.H = cv2.getPerspectiveTransform(np.float32(self.cornerCoordinates), np.float32(
                        [[0, 0], [self.BoardW, 0], [self.BoardW, self.BoardH], [0, self.BoardH]]))
                    print 'position calibration complete'
                    return
                else:
                    print 'not enough corners selected'
            elif keypress == ord('r'):
                self.cornerCount = 0
                self.cornerCoordinates = np.zeros([4, 2])
                self.calibratedBool = False
                print 'corners reset'
            elif keypress == ord('q'):
                self.release()

            for i in self.cornerCoordinates:
                cv2.circle(view, tuple(int(x) for x in i), 5, (255, 0, 0), -1)

            cv2.imshow('setup', view)

    def color_setup(self):
        canvas_filled = np.zeros((self.BoardH, self.BoardW, 3))
        canvas_blank = np.zeros((self.BoardH, self.BoardW, 3))

        cv2.imshow('Board', 255-self.canvas)
        print 'determining white background color range'
        while 1:
            ret, frame = self.vid.read()
            frame = cv2.warpPerspective(frame, self.H, (self.BoardW, self.BoardH))
            canvas_blank = np.maximum(frame, canvas_blank)
            cv2.imshow('setup', frame)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                break
            elif keypress == ord('r'):
                canvas_blank = np.zeros((self.BoardH, self.BoardW, 3))
                print 'color range reset'
            elif keypress == ord('q'):
                self.release()

        cv2.imshow('Board', self.canvas)
        print 'determining black background color range'
        while 1:
            ret, frame = self.vid.read()
            frame = cv2.warpPerspective(frame, self.H, (self.BoardW, self.BoardH))
            canvas_filled = np.maximum(frame, canvas_filled)
            cv2.imshow('setup', frame)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                self.dotsFinder.trainEyes(canvas_blank, canvas_filled)
                print 'color calibration complete'
                break
            elif keypress == ord('r'):
                canvas_filled = np.zeros((self.BoardW, self.BoardH))
                print 'color range reset'
            elif keypress == ord('q'):
                self.release()

    def release(self):
        self.vid.release()
        cv2.destroyAllWindows()
        quit()


class Cat:
    def __init__(self):
        self.canvas_blank = []
        self.canvas_filled = []
        self.canvas_max = []

        # status indices:
        # 0. previous time pressed/depressed
        # 1. current time
        # 2. pressed or depressed
        # 3. number of times clicked [0/1/2]
        # 4. double click delay threshold (default .5)

        self.status = [0] * 5
        self.status[4] = .5

        params = cv2.SimpleBlobDetector_Params()

        self.blobDetector = cv2.SimpleBlobDetector_create(params)

    def trainEyes(self, canvas_blank, canvas_filled):
        self.canvas_blank = canvas_blank
        self.canvas_filled = canvas_filled
        self.canvas_max = np.maximum(canvas_blank, canvas_filled) + 20

    def find(self, view, canvas):
        # curCanvas = np.where([canvas] != 1, self.canvas_blank, self.canvas_filled)
        # dt_view = (view - curCanvas > 30).any(2).astype(np.uint8)*255
        # print max((view - self.canvas_max).flatten())

        dt_view = (view > self.canvas_max).any(2)*uint8(255)

        # dots = self.blobDetector.detect(dt_view)
        dots = []

        if len(dots):
            print len(dots)
            if not self.status[2]:
                self.status[2] = True
                self.status[1] = time.time()
                if self.status[3] == 0:
                    self.status[3] = 1
                else:
                    if self.status[1] - self.status[0] < self.status[4]:
                        self.status[3] = 2
                    else:
                        self.status[3] = 1
                self.status[0] = self.status[1]
        else:
            if self.status[2]:
                self.status[2] = False
                self.status[1] = time.time()
                if self.status[3] == 2:
                    self.status[3] = 0

        return dt_view, dots, self.status


if __name__ == "__main__":
    res = 200
    lb = LaserBoard(3 * res, 4 * res)
    lb.start()
