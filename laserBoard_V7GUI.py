from __future__ import division
import cv2
import numpy as np
import time
import win32api
from Tkinter import *
import multiprocessing
import os
from LaserPosEstimator import LaserPosEstimator
import threading


class LaserBoard:
    def __init__(self, height, width, src=0):
        self.vid = cv2.VideoCapture(src)
        self.board_h = height
        self.board_w = width
        self.canvas = np.zeros((self.board_h, self.board_w), dtype=np.uint8)
        self.canvas_pos = []
        self.H = []
        self.canvas_bg = np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8)
        self.canvas_thresh_c = 1.2  # increase the threshold by constant
        self.q_frame = multiprocessing.Queue()
        self.q_key = multiprocessing.Queue()
        self.show_thread = multiprocessing.Process(target=show_loop,
                                                   args=(self.q_frame, self.q_key, [self.board_h, self.board_w]))
        self.min_dot_size = 10
        self.lpe = LaserPosEstimator()
        self.end = False
        self.state = 1
        self.app_state = 0
        self.window = Tk()

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
            # choice = raw_input()
            choice = str(self.state)

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
                self.pong_demo()
            elif choice == '7':
                self.camera_view()
            elif choice == '8':
                self.calibration_setup()
            elif choice == '9':
                self.release()

            self.canvas = np.zeros((self.board_h, self.board_w), dtype=np.uint8)

    @staticmethod
    def click(x, y):
        win32api.SetCursorPos((x, y))
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    def detector(self, image):
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        coord = []
        for cnt in contours:
            if len(cnt) > self.min_dot_size:
                temp = [cv2.moments(cnt)[x] for x in ['m10', 'm01', 'm00']]
                if temp[2] != 0:
                    coord.append((int(temp[0]/temp[2]), int(temp[1]/temp[2])))
        return coord

    def basic_draw(self):
        self.end = False
        self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
        prev_pos = []
        while 1:
            ret, view = self.vid.read()
            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find(board)
            key_points = self.detector(dt_view)
            if key_points:
                if prev_pos:
                    cv2.line(self.canvas, tuple(key_points[0]), tuple(prev_pos), 255, 2)
                    prev_pos = key_points[0]
                else:
                    prev_pos = key_points[0]
                    cv2.line(self.canvas, tuple(key_points[0]), tuple(prev_pos), 255, 2)
            else:
                prev_pos = []

            self.q_frame.put(255 - self.canvas)
            if self.end:
                cv2.destroyWindow('setup')
                self.q_frame.put(255 - np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                break
            elif self.app_state == 1:
                self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                print 'canvas cleared'
                self.app_state = 0
            cv2.waitKey(1)
            cv2.imshow('setup', self.canvas_bg)

    def mouse_fun(self):
        self.end = False
        while 1:
            ret, view = self.vid.read()

            if self.end:
                self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                break
            elif self.app_state == 1:
                self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                print('canvas cleared')
                self.app_state = 0

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find(board)

            key_points = self.detector(dt_view)
            if len(key_points) > 0:
                self.click(int(key_points[0][0]) + win32api.GetSystemMetrics(0),
                           int(key_points[0][1]) + 30)

    def target_shoot(self):
        self.end = False
        points = 0
        start_time = time.clock()
        target = ((np.random.rand(1, 1) * .9 + .05) * self.board_w,
                  (np.random.rand(1, 1) * .9 + .05) * self.board_h)
        while 1:
            ret, view = self.vid.read()

            if self.end:
                self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                break
            elif self.app_state == 1:
                target = ((np.random.rand(1, 1) * .9 + .05) * self.board_w,
                          (np.random.rand(1, 1) * .9 + .05) * self.board_h)
                points = 0
                self.app_state = 0

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))

            dt_view = self.find(board)

            target_range = np.zeros((self.board_h, self.board_w), dtype=np.uint8)
            cv2.circle(target_range, target, 10, 255, -1)

            if cv2.bitwise_and(dt_view, target_range).any():
                target = ((np.random.rand(1, 1) * .9 + .05) * self.board_w,
                          (np.random.rand(1, 1) * .9 + .05) * self.board_h)
                points += 1

            cv2.putText(target_range, str(points) + ' targets shot', (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            cv2.putText(target_range, str(int(time.clock() - start_time)) +
                        ' seconds', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            self.q_frame.put(target_range)
            # cv2.imshow('setup', dt_view)

    def pos_tracking_demo(self):
        self.end = False
        scale = np.array([200, -200])
        start = False
        dialog1 = 'please position a window width away'
        dialog2 = 'from the bottom left corner and press r'
        offset = np.array([self.board_w / 2, self.board_h / 2])
        while 1:

            ret, view = self.vid.read()
            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find(board)
            key_points = self.detector(dt_view)

            if self.end:
                self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                break
            elif self.app_state == 2:
                start = False
                screen_height = 2
                print 'please position the foci point one meter from the origin and press r'
                while 1:
                    ret, view = self.vid.read()
                    board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
                    dt_view = self.find(board)
                    key_points = self.detector(dt_view)
                    if self.app_state == 1:
                        break
                        
                    if len(key_points) == 3:
                        self.lpe.calibrate_angles(key_points[0], key_points[1],
                                                  key_points[2], screen_height, self.board_h)
                        start = True
                        self.app_state = 0
                        print "started detection"
                        break

            if start and (len(key_points) == 3):
                est_pos, order = self.lpe.getPos(key_points[0], key_points[1], key_points[2])
                dialog1 = 'you are standing at position:'
                dialog2 = str(est_pos)
                cv2.circle(dt_view, tuple((np.multiply(np.array(est_pos[0:2]), scale) + offset).astype(int)),
                           int(est_pos[2]*5), 255, -1)
                cv2.putText(dt_view, 'A', key_points[order[0]], cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
                cv2.putText(dt_view, 'B', key_points[order[1]], cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
                cv2.putText(dt_view, 'C', key_points[order[2]], cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)

            cv2.putText(dt_view, dialog1, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            cv2.putText(dt_view, dialog2, (10, 45), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)

            self.q_frame.put(dt_view)

    def maze_demo(self):
        self.end = False
        maze_map = cv2.resize(cv2.imread('maze.png'), (self.board_w, self.board_h))
        state = 0
        start_time = 0
        end_time = 0

        while 1:
            maze = maze_map.copy()
            ret, view = self.vid.read()

            if self.end:
                self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                break
            elif self.app_state == 1:
                self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find(board)
            pos_color = set(maze[dt_view != 0, 2])
            print(pos_color)
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
        self.end = False
        paddle_width = 80

        def hit(pad_l, pad_r, ball):
            if ball[0] < 10:
                rel_pos = pad_l[1] - ball[1]
                if abs(rel_pos) > paddle_width:
                    return None
            elif self.board_w - 10 < ball[0]:
                rel_pos = pad_r[1] - ball[1]
                if abs(rel_pos) > paddle_width:
                    return None
            else:
                return None

            ball_vel[1] += int(-rel_pos*10/paddle_width)
            if ball_vel[1] > 30:
                ball_vel[1] = 30
            ball_vel[0] = -ball_vel[0]
            return rel_pos

        pong_map = cv2.resize(cv2.imread('pong_map.png', 0), (self.board_w, self.board_h))
        begin = False
        ball_y_vel = 10
        ball_x_vel = 10
        score = [0, 0]
        ball_pos = []
        paddle_l = [10, int(self.board_h / 2)]
        paddle_r = [self.board_w - 10, int(self.board_h / 2)]
        while 1:
            ret, view = self.vid.read()

            if self.end:
                self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                break
            elif self.app_state == 1:
                score = [0, 0]
                begin = False
                ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                ball_vel = np.array([ball_x_vel, ball_y_vel])
                print('canvas cleared')

            if begin:
                ball_pos = ball_pos + ball_vel
                if hit(paddle_l, paddle_r, ball_pos):
                    pass
                elif ball_pos[0] > self.board_w:
                    begin = False
                    score[0] += 1
                    ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                    ball_vel = np.array([ball_x_vel, ball_y_vel])
                elif ball_pos[0] < 0:
                    begin = False
                    score[1] += 1
                    ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                    ball_vel = np.array([ball_x_vel, ball_y_vel])
                    ball_vel[0] = -ball_vel[0]
                elif ball_pos[1] > self.board_h or ball_pos[1] < 0:
                    ball_vel[1] = -ball_vel[1]
            else:
                ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                ball_vel = np.array([ball_x_vel, ball_y_vel])

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find(board)
            key_points = self.detector(dt_view)
            if len(key_points) > 0:
                if len(key_points) == 2:
                    begin = True
                for key_point in key_points:
                    if key_point[0] < self.board_w/2:
                        paddle_l[1] = np.array(key_point[1])
                    else:
                        paddle_r[1] = np.array(key_point[1])

            canvas = cv2.bitwise_or(pong_map.copy(), dt_view)
            cv2.rectangle(canvas, tuple(paddle_l + np.array([-10, -paddle_width])),
                          tuple(paddle_l + np.array([0, paddle_width])), 255, -1)
            cv2.rectangle(canvas, tuple(paddle_r + np.array([0, -paddle_width])),
                          tuple(paddle_r + np.array([10, paddle_width])), 255, -1)
            cv2.rectangle(canvas, tuple(ball_pos + np.array([-5, -5])), tuple(ball_pos + np.array([5, 5])), 255, -1)
            cv2.putText(canvas, str(score[0]), (10, 60), cv2.FONT_HERSHEY_PLAIN, 5, 255, 2)
            cv2.putText(canvas, str(score[1]), (self.board_w - 100, 60), cv2.FONT_HERSHEY_PLAIN, 5, 255, 2)
            self.q_frame.put(canvas)

    def camera_view(self):
        self.end = False
        while 1:
            ret, view = self.vid.read()

            if self.end:
                self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                break
            elif self.app_state == 2:
                self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))

            dt_view = self.find(board)
            key_points = self.detector(dt_view)
            print len(key_points)
            if len(key_points) > 0:
                for i in key_points:
                    cv2.circle(dt_view, i, 10, (100, 0, 0), -1)
            self.q_frame.put(255-dt_view)

    def calibration_setup(self):
        cv2.namedWindow('setup')
        self.q_frame.put(255 - np.zeros([self.board_h, self.board_w], dtype=np.uint8))
        self.position_setup()
        self.color_setup()
        cv2.destroyWindow('setup')
        self.state = 1

    def position_setup(self):
        calibration_var = [False, np.zeros([4, 2]), 0]

        def corners_clicked(event, x, y, _, calibration_stats):
            if event == cv2.EVENT_LBUTTONDOWN:
                if not calibration_stats[0]:
                    calibration_stats[1][calibration_stats[2], :] = [x, y]
                    calibration_stats[2] += 1
                    if calibration_stats[2] == 4:
                        calibration_stats[0] = True

        cv2.setMouseCallback('setup', corners_clicked, calibration_var)
        print 'calibrating screen corners please click corners from top left going clockwise'
        while 1:
            ret, view = self.vid.read()
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                if calibration_var[0]:
                    self.H = cv2.getPerspectiveTransform(np.float32(calibration_var[1]), np.float32(
                        [[0, 0], [self.board_w, 0], [self.board_w, self.board_h], [0, self.board_h]]))
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
        self.canvas_bg = np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8)

        print 'determining background color range'
        while 1:
            ret, frame = self.vid.read()
            frame = cv2.warpPerspective(frame, self.H, (self.board_w, self.board_h))
            self.canvas_bg = np.maximum(frame, self.canvas_bg)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                temp = self.canvas_bg.copy()
                self.canvas_bg = (self.canvas_bg.astype(np.float64) * self.canvas_thresh_c).astype(np.uint8)
                self.canvas_bg[temp > self.canvas_bg] = 255
                print 'color calibration complete'
                break
            elif keypress == ord('r'):
                self.canvas_bg = np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8)
                print 'color range reset'
            elif keypress == ord('q'):
                self.release()

            cv2.imshow('setup', self.canvas_bg)

    def find(self, view):
        return (view > self.canvas_bg).any(2) * np.uint8(255)

    def release(self):
        self.vid.release()
        if self.show_thread:
            self.show_thread.terminate()
        cv2.destroyAllWindows()
        self.window.destroy()
        quit()

    def in_app_change(self, num):
        if num == 1:
            self.app_state = 1
        elif num == 2:
            self.app_state = 2

    def change(self, num):
        if num == 1:
            self.state = 1
        elif num == 2:
            self.state = 2
        elif num == 3:
            self.state = 3
        elif num == 4:
            self.state = 4
        elif num == 5:
            self.state = 5
        elif num == 6:
            self.state = 6
        elif num == 7:
            self.state = 7
        elif num == 8:
            self.state = 8
        elif num == 9:
            self.state = 9
        self.end = True


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
    lb = LaserBoard(3 * res, 4 * res, 1)

    def control_window():

        lb.window.title("Laser Board Controls")
        lb.window.geometry('450x164+900+240')

        button1 = Button(lb.window, text='Exit', command=lambda: lb.change(9))
        button1.grid(row=0, column=0)
        button1.config(height=2, width=20)

        button2 = Button(lb.window, text='Mouse Functionality Test', command=lambda: lb.change(2))
        button2.grid(row=0, column=1)
        button2.config(height=2, width=20)

        button3 = Button(lb.window, text='Tracking Demo', command=lambda: lb.change(4))
        button3.grid(row=0, column=2)
        button3.config(height=2, width=20)

        button4 = Button(lb.window, text='Basic Draw', command=lambda: lb.change(1))
        button4.grid(row=1, column=0)
        button4.config(height=2, width=20)

        button5 = Button(lb.window, text='Maze', command=lambda: lb.change(5))
        button5.grid(row=1, column=1)
        button5.config(height=2, width=20)

        button6 = Button(lb.window, text='Target shoot', command=lambda: lb.change(3))
        button6.grid(row=1, column=2)
        button6.config(height=2, width=20)

        button7 = Button(lb.window, text='Pong', command=lambda: lb.change(6))
        button7.grid(row=2, column=0)
        button7.config(height=2, width=20)

        button8 = Button(lb.window, text='Recalibrate', command=lambda: lb.change(8))
        button8.grid(row=2, column=1)
        button8.config(height=2, width=20)

        button9 = Button(lb.window, text='reset', command=lambda: lb.in_app_change(1))
        button9.grid(row=2, column=2)
        button9.config(height=2, width=20)

        button10 = Button(lb.window, text='set', command=lambda: lb.in_app_change(2))
        button10.grid(row=3, column=0)
        button10.config(height=2, width=20)

        button11 = Button(lb.window, text='camera view', command=lambda: lb.change(7))
        button11.grid(row=3, column=1)
        button11.config(height=2, width=20)

        lb.window.mainloop()


    mThread = threading.Thread(target=control_window)
    mThread.daemon = True
    mThread.start()
    # lb = LaserBoard(2,1)
    lb.laser_board_run()
    mThread.join()

