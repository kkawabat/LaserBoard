from __future__ import division
import cv2
import numpy as np
import time
import win32api
import win32con
import win32gui
import multiprocessing
import os
from LaserPosEstimator import LaserPosEstimator


class LaserBoard:
    def __init__(self, height, width, src=0):
        self.vid = cv2.VideoCapture(src)
        self.BoardH = height
        self.BoardW = width
        self.canvas = np.zeros((self.BoardH, self.BoardW), dtype=np.uint8)
        self.canvas_pos = []
        self.H = []
        self.canvas_background = np.zeros((self.BoardH, self.BoardW, 3), dtype=np.uint8)
        self.canvas_thresh_c = 10
        self.q_frame = multiprocessing.Queue()
        self.q_key = multiprocessing.Queue()
        self.show_thread = multiprocessing.Process(target=show_loop,
                                                   args=(self.q_frame, self.q_key, [self.BoardH, self.BoardW]))

        self.lpe = LaserPosEstimator()

    def laser_board_run(self):
        self.show_thread.start()
        self.calibration_setup()
        while 1:
            os.system('cls')
            print 'Choose program to run'
            print '1. basic draw'
            print '2. laser mouse'
            print '3. target shooting'
            print '4. position tracking demo'
            print '5. maze demo'
            print '6. pong game'
            print '7. camera/tracking test'
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
                self.maze_demo()
            elif choice == '6':
                self.pong()
            elif choice == '7':
                self.camera_view()
            elif choice == '8':
                self.calibration_setup()
            elif choice == '9':
                self.release()

            self.canvas = np.zeros((self.BoardH, self.BoardW), dtype=np.uint8)

    @staticmethod
    def click(x, y):
        win32api.SetCursorPos((x, y))
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    @staticmethod
    def detector(image):
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        moment = [(cv2.moments(cnt)) for cnt in contours[1]]
        return [(int(round(m['m10']/m['m00'])), int(round(m['m01']/m['m00']))) for m in moment]

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

            key_points = self.detector(dt_view)
            if len(key_points) > 0:
                self.click(int(key_points[0][0]) + win32api.GetSystemMetrics(0),
                           int(key_points[0][1]) + 30)

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

            cv2.putText(target_range, str(points) + ' targets shot', (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            cv2.putText(target_range, str(int(time.clock() - start_time)) +
                        ' seconds', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            self.q_frame.put(target_range)
            cv2.imshow('setup', dt_view)

    def pos_tracking_demo(self):
        start = False
        dialog1 = ' please position a window width away'
        dialog2 = 'from the bottom left corner and press r'
        offset = np.array([self.BoardW/2, self.BoardH/2])
        while 1:
            
            ret, view = self.vid.read()
            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            dt_view = self.find(board)
            key_points = self.detector(dt_view)

            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    start = False
                    screen_height = 2
                    print 'please position the foci point one meter from the origin and press r'
                    while 1:
                        ret, view = self.vid.read()
                        board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
                        dt_view = self.find(board)
                        key_points = self.detector(dt_view)
                        if len(key_points) == 3:
                            self.lpe.calibrate_angles(key_points[0], key_points[1],
                                                      key_points[2], screen_height, self.BoardH)
                            start = True
                            print "started detection"
                            break

            if start and (len(key_points) == 3):
                est_pos = self.lpe.getPos(key_points[0], key_points[1], key_points[2])
                dialog1 = 'you are standing at position:'
                dialog2 = str(est_pos)
                cv2.circle(dt_view, int(np.array(est_pos[0:1]) + offset), int(est_pos[2]*5), 255, -1)
            cv2.putText(dt_view, dialog1, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            cv2.putText(dt_view, dialog2, (10, 45), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            self.q_frame.put(dt_view)

    def maze_demo(self):
        maze_map = cv2.resize(cv2.imread('maze.png'), (self.BoardW, self.BoardH))
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
            print pos_color
            if 0 not in pos_color:
                if 200 in pos_color and state == 0:
                    state = 1
                    start_time = time.clock()
                elif 100 in pos_color and state == 1:
                    state = 2
                    end_time = time.clock() - start_time
            else:
                state = 0
                start_time = 0

            if state == 1:
                cv2.putText(maze, "{0:.2f}".format(time.clock() - start_time) + ' seconds',
                            (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 0, 2)
            elif state == 2:
                if end_time < 6:
                    cv2.putText(maze, 'you have finished the rat race in ' + "{0:.2f}".format(end_time) +
                                ' seconds', (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 0, 2)
                else:
                    cv2.putText(maze, 'You could do better', (10, 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, 0, 2)
            else:
                cv2.putText(maze, 'Start Light Gray, Go to Dark Grey, Avoid Walls', (10, 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, 0, 2)
            self.q_frame.put(maze)

    def pong_demo(self):
        def hit(pad_l, pad_r, ball):
            if ball[0] < 10:
                if pad_l[1] + 10 >= ball[1] >= 10 - pad_l[1]:
                    return True
                else:
                    return False
            elif ball[0] > self.BoardW - 10:
                if pad_r[1] + 10 >= ball[1] >= 10 - pad_r[1]:
                    return True
                else:
                    return False
            return False

        pong_map = cv2.resize(cv2.imread('maze.png'), (self.BoardW, self.BoardH))
        begin = False
        paddle_l = [10, int(self.BoardH/2)]
        paddle_r = [self.BoardW - 10, int(self.BoardH/2)]
        while 1:
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    break
                elif keypress == ord('r'):
                    score = [0, 0]
                    begin = False
                    print('canvas cleared')

            if begin:
                ball_pos = ball_pos + ball_vel
                if hit(paddle_l, paddle_r, ball_pos):
                    ball_vel[0] = -ball_vel[0]
                elif ball_pos[0] > self.BoardW:
                    begin = False
                    score[0] += 1
                elif ball_pos[0] < 0:
                    begin = False
                    score[1] += 1
                elif ball_pos[1] > self.BoardH or ball_pos[1] < 0:
                    ball_vel[1] = -ball_vel[1]
            else:
                ball_pos = np.array([int(self.BoardW/2), int(self.BoardH/2)])
                ball_vel = np.array([5, 5])
                if score[0] < score[1]:
                    ball_vel = -ball_vel

            board = cv2.warpPerspective(view, self.H, (self.BoardW, self.BoardH))
            dt_view = self.find(board)
            key_points = self.detector(dt_view)
            if len(key_points) == 2:
                begin = True
                if key_points[0][0] < key_points[1][0]:
                    paddle_l[1] = key_points[0][1]
                    paddle_r[1] = key_points[1][1]
                else:
                    paddle_l[1] = key_points[1][1]
                    paddle_r[1] = key_points[0][1]

            cv2.rectangle(dt_view,paddle_l + [-10, 10], paddle_l + [-10, 0], 255)
            cv2.rectangle(dt_view,paddle_r + [0, 10], paddle_r + [10, -10], 255)
            cv2.rectangle(dt_view,ball_pos + [-2, -2], ball_pos + [2, 2], 255)
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
            key_points = self.detector(dt_view)
            if len(key_points) > 0:
                for i in key_points:
                    cv2.circle(dt_view, i, 2, (255, 0, 0), -1)
            self.q_frame.put(dt_view)

    def calibration_setup(self):
        cv2.namedWindow('setup')
        self.q_frame.put(255 - np.zeros([self.BoardH, self.BoardW], dtype=np.uint8))
        self.position_setup()
        self.color_setup()
        cv2.destroyWindow('setup')

    def position_setup(self):
        calibration_var = [False, np.zeros([4, 2]), 0]

        def corners_clicked(event, x, y, flag, calibration_var):
            if event == cv2.EVENT_LBUTTONDOWN:
                if not calibration_var[0]:
                    calibration_var[1][calibration_var[2], :] = [x, y]
                    calibration_var[2] += 1
                    if calibration_var[2] == 4:
                        calibration_var[0] = True

        cv2.setMouseCallback('setup', corners_clicked, calibration_var)
        print 'calibrating screen corners please click corners from top left going clockwise'
        while 1:
            ret, view = self.vid.read()

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                if calibration_var[0]:
                    self.H = cv2.getPerspectiveTransform(np.float32(calibration_var[1]), np.float32(
                        [[0, 0], [self.BoardW, 0], [self.BoardW, self.BoardH], [0, self.BoardH]]))
                    print 'position calibration complete'
                    self.canvas_pos = calibration_var[1]
                    return
                else:
                    print 'not enough corners selected'
            elif keypress == ord('r'):
                calibration_var = [False, np.zeros([4, 2]), 0]
                cv2.setMouseCallback('setup', corners_clicked, calibration_var)
                print 'corners reset'
            elif keypress == ord('q'):
                self.release()

            for i in calibration_var[1]:
                cv2.circle(view, tuple(int(x) for x in i), 2, (255, 0, 0), -1)

            cv2.imshow('setup', view)

    def color_setup(self):
        self.canvas_background = np.zeros((self.BoardH, self.BoardW, 3), dtype=np.uint8)

        print 'determining background color range'
        while 1:
            ret, frame = self.vid.read()
            frame = cv2.warpPerspective(frame, self.H, (self.BoardW, self.BoardH))
            self.canvas_background = np.maximum(frame, self.canvas_background)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                temp = self.canvas_background.copy()
                self.canvas_background += self.canvas_thresh_c
                self.canvas_background[temp > self.canvas_background] = 255

                print 'color calibration complete'
                break
            elif keypress == ord('r'):
                self.canvas_background = np.zeros((self.BoardH, self.BoardW, 3), dtype=np.uint8)
                print 'color range reset'
            elif keypress == ord('q'):
                self.release()

            cv2.imshow('setup', self.canvas_background)

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
    lb = LaserBoard(3 * res, 4 * res, 0)
    # lb = LaserBoard(2,1)
    lb.laser_board_run()
