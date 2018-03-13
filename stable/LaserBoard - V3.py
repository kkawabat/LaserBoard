import cv2
import numpy as np
import time
import win32api
import win32con


class LaserBoard:
    def __init__(self, board_height, board_width):
        self.calibratedBool = False
        self.cornerCoordinates = np.zeros([4, 2])
        self.cornerCount = 0
        self.dotFinder = Cat()
        self.vid = cv2.VideoCapture(2)
        self.BoardH = board_height
        self.BoardW = board_width
        self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
        self.dotsFinder = Cat()
        self.H = []
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.maxArea = 1000
        self.params.minArea = .1
        self.params.filterByColor = False
        self.params.minDistBetweenBlobs = 5

        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def start(self):
        # try:
            cv2.namedWindow('Board')
            cv2.namedWindow('setup')
            self.calibration_setup()
            self.laser_board_run()
        # except Exception, e:
        #    print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        #    print str(e)
            self.release()

    def laser_board_run(self):
        while 1:

            print 'Choose program to run'
            print '1. basic draw'
            print '2. mouse functionality test (not working)'
            print '3. target shooting'
            print '4. position tracking demo (not working)'
            print '5. camera view'
            print '9. quit'
            choice = input()
            if type(choice) == 'str':
                continue
            elif choice == 1:
                self.basic_draw()
            elif choice == 2:
                self.mouse_fun()
            elif choice == 3:
                self.target_shoot()
            elif choice == 4:
                self.pos_tracking_demo()
            elif choice == 5:
                self.camera_view()
            elif choice == 6:
                self.pos_track_demo()
            elif choice == 9:
                self.release()

    def click( x, y):
        win32api.SetCursorPos(( x, y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    def basic_draw(self):
        #a = 0
        while 1:
            ret, view = self.vid.read()
            #a = time.clock()
            keypress = cv2.waitKey(1) & 0xFF
            #print time.clock() - a
            if keypress == ord('q'):
                break
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

#            cv2.imshow('setup', cat_view)
            cv2.imshow('Board', 255 - self.canvas)

    def mouse_fun(self):
        while 1:
            ret, view = self.vid.read()
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'):
                break
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
                cur_status = 'doubleClicked'
            elif status[3] == 1 and status[2]:
                cur_status = 'clicked'
            else:
                cur_status = 'unpressed'

            if dots:
                curPos = (dots[0].pts[:])
            else:
                curPos = (20,10)

            cv2.putText(cat_view, str(status[1]-status[0]) + ' seconds', (10,10), cv2.FONT_HERSHEY_PLAIN, 10)
            cv2.putText(cat_view, cur_status, (curPos), cv2.FONT_HERSHEY_PLAIN, 10)
            cv2.imshow('setup', cat_view)

    def target_shoot(self):
        points = 0
        start_time = time.clock()
        target = (np.random.rand(1,1) * self.BoardW, np.random.rand(1,1) * self.BoardH)
        while 1:
            ret, view = self.vid.read()
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'):
                break
            elif keypress == ord('r'):
                target = (np.random.rand(1,1) * self.BoardW, np.random.rand(1,1) * self.BoardH)
                points = 0
            elif keypress == ord('c'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                self.calibration_setup()

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))

            dt_view, dots, status = self.dotsFinder.find(board, self.canvas)

            target_range = self.canvas.copy()
            cv2.circle(target_range, target, 10, 255, -1)

            if cv2.bitwise_and(dt_view, target_range).any():
                target = (np.random.rand(1,1) * self.BoardW, np.random.rand(1,1) * self.BoardH)
                points = points + 1



            cv2.putText(target_range, str(points) + ' targets shot', (30,30), cv2.FONT_HERSHEY_PLAIN, 3, (255))

            cv2.imshow('Board', target_range)
            cv2.imshow('setup', dt_view)

    def pos_tracking_demo(self):
        while 1:
            ret, view = self.vid.read()
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('q'):
                break
            elif keypress == ord('r'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                print('canvas cleared')
            elif keypress == ord('c'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                self.calibration_setup()

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            
            dt_view, dots, status = self.dotsFinder.find(board, self.canvas)
            dt_view = cv2.GaussianBlur(dt_view,(5,5),1)
            keypoints = self.detector.detect(dt_view)

            if len(keypoints) > 0:
                for i in range(0, len(keypoints)):
                    x = '%.1f' % keypoints[i].pt[0]
                    y = '%.1f' % keypoints[i].pt[1]
                    diameter = keypoints[i].size
                    radius = diameter/2
                    area = np.pi * (np.power(radius, 2)) #pi * r^2
                    print("Blob detected at ( " + str(x)+ " , "+ str(y) + ")" + "area = " + str(area) + " pixels")
            dt_view = cv2.drawKeypoints(dt_view, keypoints,np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow('setup', dt_view)
            # self.canvas = np.maximum(self.canvas, dt_view)

    def camera_view(self):
        #a = 0
        while 1:
            ret, view = self.vid.read()
            #a = time.clock()
            keypress = cv2.waitKey(1) & 0xFF
            #print time.clock() - a
            if keypress == ord('q'):
                break
            elif keypress == ord('r'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                print('canvas cleared')
            elif keypress == ord('c'):
                self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                self.calibration_setup()

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))

            dt_view, dots, status = self.dotsFinder.find(board, self.canvas)

            cv2.imshow('Board', dt_view)
            cv2.imshow('setup', view)

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
                self.dotsFinder.train_eyes(canvas_blank, canvas_filled)
                print 'color calibration complete'
                break
            elif keypress == ord('r'):
                canvas_filled = np.zeros((self.BoardW, self.BoardH, 3))
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

    def train_eyes(self, canvas_blank, canvas_filled):
        self.canvas_blank = canvas_blank
        self.canvas_filled = canvas_filled
        self.canvas_max = np.maximum(canvas_blank, canvas_filled) + 20

    def find(self, view, canvas):
        # curCanvas = np.where([canvas] != 1, self.canvas_blank, self.canvas_filled)
        # dt_view = (view - curCanvas > 30).any(2).astype(np.uint8)*255
        # print max((view - self.canvas_max).flatten())

        dt_view = (view > self.canvas_max).any(2)*np.uint8(255)

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
