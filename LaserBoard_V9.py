from __future__ import division
import cv2
import numpy as np
import time
import win32api
import win32con
# import win32gui
import multiprocessing
import os
from LaserPosOrientEstimator import LaserPosOrientEstimator
import pyaudio
import math


class LaserBoard:
    def __init__(self, height, width, src=0):
        self.vid = cv2.VideoCapture(src)
        self.board_h = height
        self.board_w = width
        self.canvas = np.zeros((self.board_h, self.board_w), dtype=np.uint8)
        self.canvas_pos = []
        self.H = []
        self.canvas_bg = np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8)
        self.canvas_thresh_c = 1.1  # increase the threshold by constant
        self.q_frame = multiprocessing.Queue()
        self.q_key = multiprocessing.Queue()
        self.show_thread = multiprocessing.Process(target=show_loop,
                                                   args=(self.q_frame, self.q_key, [self.board_h, self.board_w]))
        self.min_dot_size = 10
        self.lpoe = LaserPosOrientEstimator()

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
            print '8. theremin'
            print 'r. recalibrate'
            print 'q. quit'
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
                self.pong_demo()
            elif choice == '7':
                self.camera_view()
            elif choice == '8':
                self.theremin()
            elif choice == 'r':
                self.calibration_setup()
            elif choice == 'q':
                self.release()

            self.canvas = np.zeros((self.board_h, self.board_w), dtype=np.uint8)

    def basic_draw(self):
        self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
        prev_pos = []
        while 1:
            ret, view = self.vid.read()
            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find_dots(board)
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
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    cv2.destroyWindow('setup')
                    self.q_frame.put(255 - np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                    print 'canvas cleared'
            cv2.waitKey(1)
            cv2.imshow('setup', self.canvas_bg)

    def mouse_fun(self):
        mouse_click_delay = 1
        anchor_pos = np.array((0, 0))
        start_time = time.clock()
        while 1:
            ret, view = self.vid.read()

            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                    print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find_dots(board)

            key_points = self.detector(dt_view)
            if len(key_points) > 0:
                pos = np.array((int(key_points[0][0]) + win32api.GetSystemMetrics(0), int(key_points[0][1]) + 30))
                win32api.SetCursorPos(pos)
                if np.linalg.norm(pos - anchor_pos) > 30:
                    anchor_pos = pos
                    start_time = time.clock()
                else:
                    if time.clock() - start_time > mouse_click_delay:
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, pos[0], pos[1], 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, pos[0], pos[1], 0, 0)
                        start_time = time.clock()
            else:
                start_time = time.clock()

    def target_shoot(self):
        points = 0
        start_time = time.clock()
        target = ((np.random.rand(1, 1) * .9 + .05) * self.board_w,
                  (np.random.rand(1, 1) * .9 + .05) * self.board_h)
        while 1:
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
                elif keypress == ord('r'):
                    target = ((np.random.rand(1, 1) * .9 + .05) * self.board_w,
                              (np.random.rand(1, 1) * .9 + .05) * self.board_h)
                    points = 0

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))

            dt_view = self.find_dots(board)

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
        scale = np.array([200, -200])
        start = False
        dialog1 = 'please position a window width away'
        dialog2 = 'from the bottom left corner and press r'
        offset = np.array([self.board_w / 2, self.board_h / 2])
        while 1:

            ret, view = self.vid.read()
            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find_dots(board)
            key_points = self.detector(dt_view)

            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
                elif keypress == ord('r'):
                    start = False
                    screen_height = 2
                    print 'please position the foci point one meter from the origin and press r'
                    while 1:
                        ret, view = self.vid.read()
                        board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
                        dt_view = self.find_dots(board)
                        key_points = self.detector(dt_view)
                        if len(key_points) == 3:
                            self.lpoe.calibrate_angles(key_points[0], key_points[1],
                                                       key_points[2], screen_height, self.board_h)
                            start = True
                            print "started detection"
                            break

            if start and (len(key_points) == 3):
                est_pos, order = self.lpoe.getPos(key_points[0], key_points[1], key_points[2])
                est_orient = self.lpoe.getRPY()
                dialog1 = '[    X,     Y,     Z],[   Roll,   Pitch,   Yaw]:'
                dialog2 = str(np.round(est_pos, 2)) + ', ' + str(np.round(est_orient, 2))
                cv2.circle(dt_view, tuple((np.multiply(np.array(est_pos[0:2]), scale) + offset).astype(int)),
                           int(abs(est_pos[2]) * 5), 255, -1)
                cv2.putText(dt_view, 'A', key_points[order[0]], cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
                cv2.putText(dt_view, 'B', key_points[order[1]], cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
                cv2.putText(dt_view, 'C', key_points[order[2]], cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)

            cv2.putText(dt_view, dialog1, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)
            cv2.putText(dt_view, dialog2, (10, 45), cv2.FONT_HERSHEY_PLAIN, 2, 255, 2)

            self.q_frame.put(dt_view)

    def maze_demo(self):
        maze_map = cv2.resize(cv2.imread('maze.png'), (self.board_w, self.board_h))
        state = 0
        start_time = 0
        end_time = 0
        prev_pos = []

        while 1:
            maze = maze_map.copy()
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                    print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find_dots(board)
            pos_color = set(maze_map[dt_view != 0, 2])

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
                key_points = self.detector(dt_view)
                if key_points:
                    if prev_pos:
                        cv2.line(self.canvas, tuple(key_points[0]), tuple(prev_pos), 255, 2)
                        prev_pos = key_points[0]
                    else:
                        prev_pos = key_points[0]
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
                prev_pos = []
            self.q_frame.put(maze)

    def pong_demo(self):
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

            ball_vel[1] += int(-rel_pos * 20 / paddle_width)
            if ball_vel[1] > 30:
                ball_vel[1] = 30
            ball_vel[0] = -ball_vel[0]
            return rel_pos

        pong_map = cv2.resize(cv2.imread('pong_map.png', 0), (self.board_w, self.board_h))
        begin = False
        vel = [15, 10]
        score = [0, 0]
        ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
        paddle_l = [10, int(self.board_h / 2)]
        paddle_r = [self.board_w - 10, int(self.board_h / 2)]
        while 1:
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
                elif keypress == ord('r'):
                    score = [0, 0]
                    begin = False
                    ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                    ball_vel = np.array(vel)
                    print('canvas cleared')

            if begin:
                ball_pos = ball_pos + ball_vel
                if hit(paddle_l, paddle_r, ball_pos):
                    pass
                elif ball_pos[0] > self.board_w:
                    begin = False
                    score[0] += 1
                    ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                    ball_vel = np.array(vel)
                elif ball_pos[0] < 0:
                    begin = False
                    score[1] += 1
                    ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                    ball_vel = np.array(vel)
                    ball_vel[0] = -ball_vel[0]
                elif ball_pos[1] > self.board_h or ball_pos[1] < 0:
                    ball_vel[1] = -ball_vel[1]
            else:
                ball_pos = np.array([int(self.board_w / 2), int(self.board_h / 2)])
                ball_vel = np.array(vel)

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find_dots(board)
            key_points = self.detector(dt_view)
            if len(key_points) > 0:
                if len(key_points) == 2:
                    begin = True
                for key_point in key_points:
                    if key_point[0] < self.board_w / 2:
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
        while 1:
            ret, view = self.vid.read()
            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
                elif keypress == ord('r'):
                    self.canvas = np.zeros([self.board_h, self.board_w], dtype=np.uint8)
                    print('canvas cleared')

            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))

            dt_view = self.find_dots(board)
            key_points = self.detector(dt_view)
            print len(key_points)
            if len(key_points) > 0:
                for i in key_points:
                    cv2.circle(dt_view, i, 10, (100, 0, 0), -1)
            self.q_frame.put(255 - dt_view)

    def theremin(self):

        def sine(f, a, length, rate):
            length = int(length * rate)
            factor = float(f) * (math.pi * 2) / rate
            return np.sin(np.arange(length) * factor) * a

        theremin = cv2.resize(cv2.imread('theremin.png'), (self.board_w, self.board_h))

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=44100,
                        output=True)

        while 1:
            ret, view = self.vid.read()
            board = cv2.warpPerspective(view, self.H, (self.board_w, self.board_h))
            dt_view = self.find_dots(board)
            keypoints = self.detector(dt_view)

            if not self.q_key.empty():
                keypress = self.q_key.get_nowait()
                if keypress == ord('q'):
                    self.q_frame.put(np.zeros([self.board_h, self.board_w], dtype=np.uint8))
                    return
            print keypoints
            if len(keypoints) == 2:
                keypoints = sorted(keypoints, key=lambda x: x[0])
                in_amp_range = ((0.1 * self.board_w < keypoints[0][0] < 0.30 * self.board_w) and
                                (0.2 * self.board_h < keypoints[0][1] < 0.75 * self.board_h))
                in_freq_range = ((0.4 * self.board_w < keypoints[1][0] < 0.9 * self.board_w) and
                                 (0.1 * self.board_h < keypoints[1][1] < 0.75 * self.board_h))
                if in_amp_range and in_freq_range:
                    freq = 2 * 10**((abs(keypoints[1][0] - (self.board_w * .9)) / (.5*self.board_w))*3 + 1)
                    amp = (abs(keypoints[0][1] - (self.board_h * .75)) / (.65*self.board_w))*2
                    chunk = sine(freq, amp, .2, 44100)
                    stream.write(chunk.astype(np.float32).tostring())
                else:
                    chunk = np.array([])
            self.q_frame.put(theremin)

    def calibration_setup(self):
        cv2.namedWindow('setup')

        self.position_setup()
        self.color_setup()
        cv2.destroyWindow('setup')

    def position_setup(self):
        self.q_frame.put(cv2.resize(cv2.imread('marker_map.png', 0), (self.board_w, self.board_h)))
        mkr_x = self.board_w * (68. / 600)
        mkr_y = self.board_h * (68. / 800)
        # Initiate SIFT detector
        try:
            orb = cv2.ORB()
        except:
            orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        marker = cv2.imread('marker.png', 0)  # (center: 70 pixels)      # queryImage

        calibration_var = [False, np.zeros([4, 2]), 0]
        bucket = []
        automated = True

        def corners_clicked(event, x, y, _, calibration_stats):
            if event == cv2.EVENT_LBUTTONDOWN:
                if not calibration_stats[0]:
                    calibration_stats[1][calibration_stats[2], :] = [x, y]
                    calibration_stats[2] += 1
                    if calibration_stats[2] == 4:
                        calibration_stats[0] = True

        cv2.setMouseCallback('setup', corners_clicked, calibration_var)

        print 'calibrating screen corners please click on markers from top left going clockwise'
        while 1:
            ret, view = self.vid.read()
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord('e'):
                if calibration_var[0]:
                    if automated:
                        self.H = cv2.getPerspectiveTransform(np.float32(calibration_var[1]),
                                                             np.float32([[mkr_x, mkr_y],
                                                                         [-mkr_x + self.board_w, mkr_y],
                                                                         [-mkr_x + self.board_w, -mkr_y + self.board_h],
                                                                         [mkr_x, -mkr_y + self.board_h]]))
                    else:
                        mkr_x = 0
                        mkr_y = 0
                        self.H = cv2.getPerspectiveTransform(np.float32(calibration_var[1]),
                                                             np.float32([[mkr_x, mkr_y],
                                                                         [-mkr_x + self.board_w, mkr_y],
                                                                         [-mkr_x + self.board_w, -mkr_y + self.board_h],
                                                                         [mkr_x, -mkr_y + self.board_h]]))
                    print 'position calibration complete'
                    self.canvas_pos = calibration_var[1]
                    self.q_frame.put(255 - np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8))
                    return
                else:
                    print 'not enough corners selected'
            elif keypress == ord('r'):
                calibration_var = [False, np.zeros([4, 2]), 0]
                cv2.setMouseCallback('setup', corners_clicked, calibration_var)
                print 'corners reset'
            elif keypress == ord('q'):
                self.release()
            elif keypress == ord('i'):
                automated = not automated
                calibration_var = [False, np.zeros([4, 2]), 0]
                cv2.setMouseCallback('setup', corners_clicked, calibration_var)

            if automated:
                # find the keypoints and descriptors with SIFT
                kp1, des1 = orb.detectAndCompute(marker, None)
                kp2, des2 = orb.detectAndCompute(view, None)

                if des1 is not None and des2 is not None:
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    if len(matches) > 10:
                        for x in matches[:10]:
                            bucket.append(np.array(kp2[x.trainIdx].pt))
                    if len(bucket) > 200:
                        bucket = bucket[-200:]
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                        _, _, centers = cv2.kmeans(np.float32(bucket), 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                        calibration_var[0] = True
                        temp = np.array(sorted(centers, key=lambda x: x[0]), dtype=int)
                        temp = sorted(temp[0:2], key=lambda x: x[1]) + sorted(temp[2:4], key=lambda x: x[1])
                        calibration_var[1] = [temp[0], temp[2], temp[3], temp[1]]
                        calibration_var[2] = 4

            for i in calibration_var[1]:
                cv2.circle(view, tuple(int(x) for x in i), 2, (255, 0, 0), -1)

            cv2.imshow('setup', view)

    def color_setup(self):
        def nothing(_):
            pass

        cv2.createTrackbar('threshold', 'setup', 0, 200, nothing)
        cv2.setTrackbarPos('threshold', 'setup', 110)
        cv2.resizeWindow('setup', self.board_w, self.board_h)
        temp_canvas_bg = np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8)

        max_frames_grabbed = 0
        while 1:
            max_frames_grabbed += 20
            print 'estimating max background intensity'
            for i in range(1, max_frames_grabbed):
                ret, frame = self.vid.read()
                frame = cv2.warpPerspective(frame, self.H, (self.board_w, self.board_h))
                temp_canvas_bg = np.maximum(frame, temp_canvas_bg)
                time.sleep(.1)

            print 'please choose optimal thresholding value (remove noise but keep laser dots)'
            while 1:
                ret, frame = self.vid.read()
                frame = cv2.warpPerspective(frame, self.H, (self.board_w, self.board_h))

                temp_thresh = cv2.getTrackbarPos('threshold', 'setup') / 100

                self.canvas_bg = (temp_canvas_bg.astype(np.float64) * temp_thresh)
                self.canvas_bg[self.canvas_bg > 255] = 250

                dt_view = self.find_dots(frame)
                keypress = cv2.waitKey(1) & 0xFF
                if keypress == ord('e'):
                    print 'color calibration complete'
                    self.canvas_thresh_c = temp_thresh
                    cv2.destroyWindow('test')
                    return
                elif keypress == ord('r'):
                    temp_canvas_bg = np.zeros((self.board_h, self.board_w, 3), dtype=np.uint8)
                    print 'color range reset'
                    break
                elif keypress == ord('q'):
                    self.release()
                cv2.imshow('setup', dt_view)

    def find_dots(self, view):
        return (view > self.canvas_bg).any(2) * np.uint8(255)

    def detector(self, image):
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        coord = []
        for cnt in contours:
            if len(cnt) > self.min_dot_size:
                temp = [cv2.moments(cnt)[x] for x in ['m10', 'm01', 'm00']]
                if temp[2] != 0:
                    coord.append((int(temp[0] / temp[2]), int(temp[1] / temp[2])))
        return coord

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
    lb = LaserBoard(3 * res, 4 * res, 1)
    dots = cv2.imread('testdot1.png',0)
    lb.detector(dots)
    lb.laser_board_run()
