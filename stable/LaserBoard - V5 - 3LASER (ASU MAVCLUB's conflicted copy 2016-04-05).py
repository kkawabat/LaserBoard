from __future__ import division 
import cv2
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
from numpy.linalg import inv
from numpy.linalg import norm
import time
import win32api
import win32con
import multiprocessing
import os


class LaserBoard:
    def __init__(self, height, width, src = 0):
        self.vid = cv2.VideoCapture(src)
        self.BoardH = height
        self.BoardW = width
        self.canvas = np.zeros((self.BoardH, self.BoardW), dtype=np.uint8)
        self.canvas_pos = []
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
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

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
            if len(key_points):
                self.click(int(key_points[0].pt[0]) + self.canvas_pos[0], int(key_points[0].pt[1]) + self.canvas_pos[1])

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

        # Setup
        angle1 = 15; 
        angle2 = 14; 
        angle3 = 13;
        Xi = np.array([0,0,1])
        LaserDotScaleFactor = .01

        # Solves the Resection problem for lengths
        def resectf(X):
            global AB
            global AC
            global BC
            
            f = np.zeros(3)
            f[0] = (2*X[0]*X[1]*np.cos(np.radians(angle1))-X[0]*X[0] - X[1]*X[1] + AB*AB)
            f[1] = (2*X[0]*X[2]*np.cos(np.radians(angle2))-X[0]*X[0] - X[2]*X[2] + AC*AC)
            f[2] = (2*X[1]*X[2]*np.cos(np.radians(angle3))-X[1]*X[1] - X[2]*X[2] + BC*BC)
            return f

        # Solves the Directions Vectors of each laser
        def vectf(X):
            global AP
            global BP
            global CP
            global a
            global b
            global c
            
            f = np.zeros(9)
            f[0] = -(AP*(X[0]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - BP*(X[3]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5])))) + (a[0] - b[0])
            f[1] = -(AP*(X[1]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - BP*(X[4]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5])))) + (a[1] - b[1])
            f[2] = -(AP*(X[2]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - BP*(X[5]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5])))) 
            f[3] = -(AP*(X[0]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - CP*(X[6]/(np.sqrt(X[6]*X[6] + X[7]*X[7]  + X[8]*X[8])))) + (a[0] - c[0])
            f[4] = -(AP*(X[1]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - CP*(X[7]/(np.sqrt(X[6]*X[6] + X[7]*X[7]  + X[8]*X[8])))) + (a[1] - c[1])
            f[5] = -(AP*(X[2]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - CP*(X[8]/(np.sqrt(X[6]*X[6] + X[7]*X[7]  + X[8]*X[8])))) 
            f[6] = -(CP*(X[6]/(np.sqrt(X[6]*X[6] + X[7]*X[7] + X[8]*X[8]))) - BP*(X[3]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5])))) + (c[0] - b[0])
            f[7] = -(CP*(X[7]/(np.sqrt(X[6]*X[6] + X[7]*X[7] + X[8]*X[8]))) - BP*(X[4]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5])))) + (c[1] - b[1])
            f[8] = -(CP*(X[8]/(np.sqrt(X[6]*X[6] + X[7]*X[7] + X[8]*X[8]))) - BP*(X[5]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5]))))
            return f

        def getPos(A,B,C,XOLD):
            global AB
            global BC
            global AC
            global AP
            global BP
            global CP
            global a
            global b
            global c

            tic = time.clock()

            # Form Estiation of Laser Vectors
            LA0 = A - XOLD;
            LB0 = B - XOLD;
            LC0 = C - XOLD;

            # Form Guess of L0 and V0
            L0 = np.array([norm(LA0),norm(LB0),norm(LC0)])
            print('L0:')
            print(L0)
            VA0 = LA0/norm(LA0);
            VB0 = LB0/norm(LB0);
            VC0 = LC0/norm(LC0);
            V0 = np.array([VA0[0],VA0[1],VA0[2],VB0[0],VB0[1],VB0[2],VC0[0],VC0[1],VC0[2]]);
            print('V0:')
            print(V0)
            # Get Laser Points as 2D Vector
            a = A[0:2]
            b = B[0:2]
            c = C[0:2]

            print('a:')
            print(a)
            print('b:')
            print(b)
            print('c:')
            print(c)
            
            # Get Distance Between Points
            AB = norm(b-a);
            BC = norm(c-b);
            AC = norm(c-a);

            # Solve Resection for Length of Lasers
            ticL = time.clock()
            L = fsolve(resectf,L0)
            print('L:')
            print(L)
            tocL = time.clock()
            AP = L[0];
            BP = L[1];
            CP = L[2];

            # Get Direction Vectors of Lasers
            ticV = time.clock()
            V = fsolve(vectf,V0)
            print('V:')
            print(V)
            tocV = time.clock()

            # Ensure Directions are Normalized 
            V[0:3] = V[0:3]/norm(V[0:3])
            V[3:6] = V[3:6]/norm(V[3:6])
            V[6:9] = V[6:9]/norm(V[6:9])

            # Estimated Lasers
            VA = np.dot(np.transpose(V[0:3]),AP)
            VB = np.dot(np.transpose(V[3:6]),BP)
            VC = np.dot(np.transpose(V[6:9]),CP)

            # Check Angle Condition
            thetaAB = np.arccos((np.dot(np.transpose(VA),VB))/(norm(VA)*norm(VB)))
            thetaAC = np.arccos((np.dot(np.transpose(VA),VC))/(norm(VA)*norm(VC)))
            thetaCB = np.arccos((np.dot(np.transpose(VC),VB))/(norm(VC)*norm(VB)))

            # Estimate Position from each laser
            XYZa = np.array([a[0],a[1],0]) - VA
            XYZb = np.array([b[0],b[1],0]) - VB
            XYZc = np.array([c[0],c[1],0]) - VC

            # Average Position Estimation
            X = (XYZa + XYZb + XYZc)/3;
            print('X Solution:')
            print(X)

            #outputs
            f = np.zeros(33)
            f[0:3] = X #Solution Position
            f[3:6] = L0 #Estimated Lengths
            f[6:9] = L #Solution Lengths
            f[9:18] = V0 #Estimated Direction Vectors
            f[18:27] = V #Solution Direction Vectors
            f[27] = thetaAB #angle1 condition
            f[28] = thetaAC #angle2 condition
            f[29] = thetaCB #angle3 condition
            f[30] = tocL - ticL # time elapsed for L
            f[31] = tocV - ticV # time elapsed for V
            toc = time.clock()
            f[32] = toc - tic # total time elapsed for solution
            return f

        a = 0

        # Main Loop
        HasInitialPosition = 'false'
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

            if HasInitialPosition == False:
                if not self.q_key.empty():
                    keypress = self.q_key.get_nowait()
                    if keypress == ord('q'):
                        break
                    elif keypress == ord('e'):
                        X = Xi
                        HasInitialPosition = True

            if (HasInitialPosition == True) and (len(key_points) == 3):
                for i in range(0, len(key_points)):
                    x = '%.1f' % key_points[i,0]
                    y = '%.1f' % key_points[i,1]
                    diameter = key_points[i].size
                    print("Blob " + str(i) + " detected at ( " + str(x) + " , " + str(y) + ")" +
                          "size = " + str(diameter) + " pixels wide")
                                        (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                A = np.array([key_points[0,0],key_points[0,1],0])
                B = np.array([key_points[1,0],key_points[1,1],0])
                C = np.array([key_points[2,0],key_points[2,1],0]) 
                SolutionData = getPos(A,B,C,X)
                

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
                            (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 0)
            elif state == 2:
                if end_time < 6:
                    cv2.putText(maze, 'you have finished the rat race in ' + "{0:.2f}".format(end_time) +
                                ' seconds', (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, 0)
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
                print 'corners reset'
            elif keypress == ord('q'):
                self.release()

            for i in calibration_var[1]:
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
    lb = LaserBoard(3 * res, 4 * res, 2)
    # lb = LaserBoard(2,1)
    lb.laser_board_run()
