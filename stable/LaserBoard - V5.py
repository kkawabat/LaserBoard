import cv2
import numpy as np
import time
import win32api
import win32con
import multiprocessing
import os



class LaserBoard:
    def __init__(self, board_height, board_width):
        self.calibratedBool = False
        self.cornerCoordinates = np.zeros([4, 2])
        self.cornerCount = 0
        self.vid = cv2.VideoCapture(2)
        self.BoardH = board_height
        self.BoardW = board_width
        self.canvas = self.canvas_background = np.zeros((self.BoardH, self.BoardW), dtype=np.uint8)
        self.H = []
        self.canvas_background = np.zeros((self.BoardH, self.BoardW, 3), dtype=np.uint8)
        self.q_frame = multiprocessing.Queue()
        self.q_key = multiprocessing.Queue()
        self.show_thread = multiprocessing.Process(target=show_loop,
                                                   args=(self.q_frame, self.q_key, [self.BoardH, self.BoardW]))

        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.maxArea = 1000
        self.params.minArea = .1
        self.params.filterByColor = False
        self.params.minDistBetweenBlobs = 5
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def laser_board_run(self):
        self.show_thread.start()
        self.calibration_setup()
        while 1:
            os.system('cls')
            print 'Choose program to run'
            print '1. basic draw'
            print '2. mouse functionality test'
            print '3. target shooting'
            print '4. position tracking demo'
            print '5. camera view'
            print '6. maze demo'
            print '8. recalibrate'
            print '9. quit'
            choice = raw_input()

            if choice == '1':
                self.basic_draw()
            elif choice == '2':
                self.mouse_fun()
            elif choice == '3':
                self.target_shoot()
            elif choice == '4':
                self.pos_tracking_demo()
            elif choice == '5':
                self.camera_view()
            elif choice == '6':
                self.maze_demo()
            elif choice == '8':
                self.calibration_setup()
            elif choice == '9':
                self.release()

    @staticmethod
    def click(x, y):
        win32api.SetCursorPos((x, y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    def basic_draw(self):
        self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)

        while 1:
            ret, view = self.vid.read()
            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            dt_view = self.find(board)
            self.canvas = np.maximum(self.canvas, dt_view)
            self.q_frame.put(255 - self.canvas)
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                print keypress
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                    print 'canvas cleared'
            cv2.waitKey(1)
            cv2.imshow('setup', self.canvas_background)

    def mouse_fun(self):
        while 1:
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                    print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            dt_view = self.find(board)
            key_points = self.detector.detect(dt_view)

    def target_shoot(self):
        points = 0
        start_time = time.clock()
        target = ((np.random.rand(1, 1) * .9 + .05) * self.BoardW,
                  (np.random.rand(1, 1) * .9 + .05) * self.BoardH)
        while 1:
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    target = ((np.random.rand(1, 1) * .9 + .05) * self.BoardW,
                              (np.random.rand(1, 1) * .9 + .05) * self.BoardH)
                    points = 0

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))

            dt_view = self.find(board)

            target_range = np.zeros((self.BoardH, self.BoardW), dtype=np.uint8)
            cv2.circle(target_range, target, 10, 255, -1)

            if cv2.bitwise_and(dt_view, target_range).any():
                target = ((np.random.rand(1, 1) * .9 + .05) * self.BoardW,
                          (np.random.rand(1, 1) * .9 + .05) * self.BoardH)
                points += 1

            cv2.putText(target_range, str(points) + ' targets shot', (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 255)
            cv2.putText(target_range, str(int(time.clock() - start_time)) +
                        ' seconds', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, 255)
            self.q_frame.put(target_range)
            cv2.imshow('setup', dt_view)

    def pos_tracking_demo(self):
        a = 0
        while 1:
            print time.clock() - a
            a = time.clock()
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                    print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            dt_view = cv2.GaussianBlur(self.find(board), (5, 5), 1)
            key_points = self.detector.detect(dt_view)

            if len(key_points) > 0:
                for i in range(0, len(key_points)):
                    x = '%.1f' % key_points[i].pt[0]
                    y = '%.1f' % key_points[i].pt[1]
                    diameter = key_points[i].size
                    print("Blob detected at ( " + str(x) + " , " + str(y) + ")" +
                          "size = " + str(diameter) + " pixels wide")
            dt_view = cv2.drawKeypoints(dt_view, key_points, np.array([]),
                                        (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            self.q_frame.put(dt_view)

    def camera_view(self):
        while 1:
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                    print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))

            dt_view = self.find(board)

            self.q_frame.put(dt_view)

    def maze_demo(self):
        maze_map = cv2.imread('maze.png')
        state = 0
        start_time = 0
        end_time = 0

        while 1:
            maze = maze_map.copy()
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.BoardH, self.BoardW], dtype=np.uint8)
                    print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            dt_view = self.find(board)
            pos_color = set(maze[dt_view != 0, 2])

            if 0 not in pos_color:
                print pos_color
                if 200 in pos_color and state == 0:
                    state = 1
                    start_time = time.clock()
                elif 100 in pos_color and state == 1:
                    state = 2
                    end_time = time.clock() - start_time
            else:
                print 0
                state = 0
                start_time = 0

            if state == 1:
                cv2.putText(maze, "{0:.2f}".format(time.clock() - start_time) + ' seconds',
                            (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 0)
            elif state == 2:
                if end_time < 6:
                    cv2.putText(maze, 'you have finished the rat race', (10, 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, 0)
                else:
                    cv2.putText(maze, 'You could do better', (10, 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, 0)
            else:
                cv2.putText(maze, 'Start Light Gray, Go to Dark Grey, Avoid Walls', (10, 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, 0)
            self.q_frame.put(maze)

    def calibration_setup(self):
        cv2.namedWindow('setup')
        self.q_frame.put(255 - np.zeros([self.BoardH, self.BoardW], dtype=np.uint8))
        self.position_setup()
        self.color_setup()

    def corners_clicked(self, event, x, y, *_):
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
                cv2.circle(view, tuple(int(x) for x in i), 2, (255, 0, 0), -1)

            cv2.imshow('setup', view)

    def color_setup(self):
        self.canvas_background = np.zeros((self.BoardH, self.BoardW, 3), dtype=np.uint8)

        print 'determining white background color range'
        while 1:
            ret, frame = self.vid.read()
            frame = cv2.warpPerspective(frame, self.H, (self.BoardW, self.BoardH))
            self.canvas_background = np.maximum(frame, self.canvas_background)
            keypress = cv2.waitKey(1) & 0xFF
            cv2.imshow('setup', self.canvas_background)
            if keypress == ord('e'):
                temp = self.canvas_background.copy()
                self.canvas_background += 20
                self.canvas_background[temp > self.canvas_background] = 255

                print 'color calibration complete'
                break
            elif keypress == ord('r'):
                self.canvas_background = np.zeros((self.BoardH, self.BoardW, 3), dtype=np.uint8)
                print 'color range reset'
            elif keypress == ord('q'):
                self.release()

    def find(self, view):
        return (view > self.canvas_background).any(2)*np.uint8(255)

    def release(self):
        self.vid.release()
        if self.show_thread:
            self.show_thread.terminate()
        cv2.destroyAllWindows()
        quit()


def show_loop(q_frame, q_key, dim):
    cv2.namedWindow('Board')
    from_queue = []
    while 1:
        keypress = cv2.waitKey(1) & 0xFF
        if keypress != 255:
            q_key.put(keypress)
        if not q_frame.empty():
            from_queue = q_frame.get_nowait()
            cv2.imshow('Board', from_queue)
        else:
            if len(from_queue) == 0:
                cv2.imshow('Board', 255 - np.zeros(dim, dtype=np.uint8))

if __name__ == "__main__":
    res = 200
    lb = LaserBoard(3 * res, 4 * res)
    lb.laser_board_run()
