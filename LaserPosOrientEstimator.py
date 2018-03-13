from __future__ import division 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
import time
from numpy.linalg import inv
from numpy.linalg import norm


class LaserPosOrientEstimator:
    def __init__(self, angles=(15, 14, 13)):
        self.angles = np.array(angles)
        self.prev_pos = np.array([0, 0, 2])
        self.screen_height = 2
        self.board_h = 0
        self.orient_mat = []
        self.L = []
        self.AB = []
        self.BC = []
        self.CA = []
        self.A = []
        self.B = []
        self.C = []
        self.Ao = []
        self.Bo = []
        self.Co = []
        self.VA = []
        self.VB = []
        self.VC = []

    def calibrate_angles(self, A, B, C, screen_height, boardH):
        self.board_h = boardH
        self.screen_height = screen_height
        self.prev_pos = np.array([0, 0, 2])
        self.A, self.B, self.C = self.convert_keypoints(A, B, C)
        self.AB, self.BC, self.CA = norm(self.B - self.A), norm(self.C - self.B), norm(self.C - self.A)
        X = np.array([norm(self.prev_pos - self.A), norm(self.prev_pos - self.B), norm(self.prev_pos - self.C)])
        self.angles[0] = np.degrees(np.arccos((np.square(X[0]) + np.square(X[1]) -
                                               np.square(self.AB))/(2*X[0]*X[1])))
        self.angles[1] = np.degrees(np.arccos((np.square(X[0]) + np.square(X[2]) -
                                               np.square(self.CA))/(2*X[0]*X[2])))
        self.angles[2] = np.degrees(np.arccos((np.square(X[1]) + np.square(X[2]) -
                                               np.square(self.BC))/(2*X[1]*X[2])))
        self.Ao,self.Bo,self.Co = self.A, self.B, self.C

        LA = self.A - self.prev_pos
        LB = self.B - self.prev_pos
        LC = self.C - self.prev_pos

        self.orient_mat = np.matrix([[LA[0], LA[1], LA[2], 0,     0,     0,    0,     0,     0],
             [0,     0,     0,     LA[0], LA[1], LA[2], 0,    0,     0],
             [0,     0,     0,     0,    0,     0,     LA[0], LA[1], LA[2]],
             [LB[0], LB[1], LB[2], 0,    0,    0,    0,     0,    0],
             [0,    0,    0,    LB[0], LB[1], LB[2], 0,     0,     0],
             [0,    0,    0,    0,   0,    0,    LB[0], LB[1], LB[2]],
             [LC[0], LC[1], LC[2], 0,    0,    0,     0,     0,    0],
             [0,    0,    0,     LC[0], LC[1], LC[2], 0,     0,     0],
             [0,    0,    0,     0,    0,    0,    LC[0], LC[1], LC[2]]])

        print self.angles

    # Solves the Resection problem for lengths
    def length_f(self, X):

        f = np.zeros(3)
        f[0] = 2*X[0]*X[1]*np.cos(np.radians(self.angles[0])) - np.square(X[0]) - np.square(X[1]) + np.square(self.AB)
        f[1] = 2*X[0]*X[2]*np.cos(np.radians(self.angles[1])) - np.square(X[0]) - np.square(X[2]) + np.square(self.CA)
        f[2] = 2*X[1]*X[2]*np.cos(np.radians(self.angles[2])) - np.square(X[1]) - np.square(X[2]) + np.square(self.BC)
        return f

    # Solves the Directions Vectors of each laser
    def vec_f(self, X):
        f = np.zeros(9)

        X1norm = norm(X[0:3])
        X2norm = norm(X[3:6])
        X3norm = norm(X[6:9])
        f[0] = -(self.L[0]*(X[0]/X1norm) - self.L[1]*(X[3]/X2norm)) + (self.A[0] - self.B[0])
        f[1] = -(self.L[0]*(X[1]/X1norm) - self.L[1]*(X[4]/X2norm)) + (self.A[1] - self.B[1])
        f[2] = -(self.L[0]*(X[2]/X1norm) - self.L[1]*(X[5]/X2norm))
        f[3] = -(self.L[0]*(X[0]/X1norm) - self.L[2]*(X[6]/X3norm)) + (self.A[0] - self.C[0])
        f[4] = -(self.L[0]*(X[1]/X1norm) - self.L[2]*(X[7]/X3norm)) + (self.A[1] - self.C[1])
        f[5] = -(self.L[0]*(X[2]/X1norm) - self.L[2]*(X[8]/X3norm))
        f[6] = -(self.L[2]*(X[6]/X3norm) - self.L[1]*(X[3]/X2norm)) + (self.C[0] - self.B[0])
        f[7] = -(self.L[2]*(X[7]/X3norm) - self.L[1]*(X[4]/X2norm)) + (self.C[1] - self.B[1])
        f[8] = -(self.L[2]*(X[8]/X3norm) - self.L[1]*(X[5]/X2norm))

        return f

    # The main function, solves for position given three lasers
    def getPos(self, A, B, C):
        order = []
        self.A, self.B, self.C = self.convert_keypoints(A, B, C)

        # Estimate A,B,C by minimizing the distance
        D1 = norm(self.Ao-self.A) + norm(self.Bo-self.B) + norm(self.Co-self.C)
        D2 = norm(self.Ao-self.A) + norm(self.Bo-self.C) + norm(self.Co-self.B)
        D3 = norm(self.Ao-self.B) + norm(self.Bo-self.A) + norm(self.Co-self.C)
        D4 = norm(self.Ao-self.B) + norm(self.Bo-self.C) + norm(self.Co-self.A)
        D5 = norm(self.Ao-self.C) + norm(self.Bo-self.A) + norm(self.Co-self.B)
        D6 = norm(self.Ao-self.C) + norm(self.Bo-self.B) + norm(self.Co-self.A)
        Darray = D1, D2, D3, D4, D5, D6
        Dindex = np.argmin(Darray)
        if Dindex == 0:
            self.A, self.B, self.C = self.A, self.B, self.C
            order = [0, 1, 2]
        elif Dindex == 1:
            self.A, self.B, self.C = self.A, self.C, self.B
            order = [0, 2, 1]
        elif Dindex == 2:
            self.A, self.B, self.C = self.B, self.A, self.C
            order = [1, 0, 2]
        elif Dindex == 3:
            self.A, self.B, self.C = self.B, self.C, self.A
            order = [1, 2, 0]
        elif Dindex == 4:
            self.A, self.B, self.C = self.C, self.A, self.B
            order = [2, 0, 1]
        elif Dindex == 5:
            self.A, self.B, self.C = self.C, self.B, self.A
            order = [2, 1, 0]

        self.Ao, self.Bo, self.Co = self.A, self.B, self.C

        # Form Estimation of Laser Lengths
        L0 = np.array([norm(self.A - self.prev_pos), norm(self.B - self.prev_pos), norm(self.C - self.prev_pos)])

        # Form Estimation of Laser Directional Vectors
        V0 = np.array([(self.A - self.prev_pos)/L0[0], (self.B - self.prev_pos)/L0[1], (self.C - self.prev_pos)/L0[2]])
        
        # Get Distance Between Points
        self.AB, self.BC, self.CA = norm(self.B - self.A), norm(self.C - self.B), norm(self.C - self.A)

        # Solve Resection for Length of Lasers
        self.L = fsolve(self.length_f, L0)

        # Get Direction Vectors of Lasers
        V = fsolve(self.vec_f, V0)

        # Ensure Directions are Normalized
        V[0:3] = V[0:3]/norm(V[0:3])
        V[3:6] = V[3:6]/norm(V[3:6])
        V[6:9] = V[6:9]/norm(V[6:9])

        # Estimated Lasers
        self.VA = V[0:3]*self.L[0]
        self.VB = V[3:6]*self.L[1]
        self.VC = V[6:9]*self.L[2]

        # Average Position Estimation
        X = ((self.A - self.VA) + (self.B - self.VB) + (self.C - self.VC))/3

        # outputs
        self.prev_pos = X
        return X, order

    def getRPY(self):
        DV = np.array([self.VA*(1/norm(self.VA)), self.VB*(1/norm(self.VB)), self.VC*(1/norm(self.VC))]).flatten()
        r_mat = np.dot(np.linalg.pinv(self.orient_mat), DV).reshape(3,3)

        yaw1 = np.arctan2(r_mat[1, 0], r_mat[0, 0])
        pitch1 = np.arctan2(-r_mat[2, 0], np.sqrt(r_mat[2,1]*r_mat[2, 1] + r_mat[2, 2]*r_mat[2, 2]))
        pitch2 = np.arctan2(r_mat[2, 0], np.sqrt(r_mat[2, 1]*r_mat[2, 1] + r_mat[2, 2]*r_mat[2, 2]))
        roll1 = -np.arctan2(r_mat[2, 1], r_mat[2, 2])
        roll2 = np.arctan2(r_mat[2, 1], r_mat[2, 2])

        self.roll = roll1*180/np.pi
        self.pitch = pitch1*180/np.pi
        self.yaw = yaw1*180/np.pi

        return self.roll, self.pitch, self.yaw

    def convert_keypoints(self, A, B, C):
        A = np.array([A[0], self.board_h - A[1], 0])*(self.screen_height/self.board_h)
        B = np.array([B[0], self.board_h - B[1], 0])*(self.screen_height/self.board_h)
        C = np.array([C[0], self.board_h - C[1], 0])*(self.screen_height/self.board_h)
        return A,B,C
if __name__ == '__main__':
    a = LaserPosEstimator()
