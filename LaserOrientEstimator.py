from __future__ import division 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
import time
from numpy.linalg import inv
from numpy.linalg import norm


class LaserOrientEstimator:
    def __init__(self, angles=(15,14,13)):
        self.A = []

    # Get rpy angles
    def getRPY(self,VA,VB,VC):

        DV = np.matrix([VA*(1/norm(VA)),VB*(1/norm(VB)),VC*(1/norm(VC))]);
        DV = np.reshape(DV,(9,1))
        r = np.dot(np.dot(inv(np.dot(self.A.T,self.A)),self.A.T),DV)
        Rmat = np.array([[r[0], r[1], r[2]],
                         [r[3], r[4], r[5]],
                         [r[6], r[7], r[8]]])

        yaw1 = np.arctan2(Rmat[1,0],Rmat[0,0])
        pitch1 = np.arctan2(-Rmat[2,0],np.sqrt(Rmat[2,1]*Rmat[2,1] + Rmat[2,2]*Rmat[2,2]))
        pitch2 = np.arctan2(Rmat[2,0],np.sqrt(Rmat[2,1]*Rmat[2,1] + Rmat[2,2]*Rmat[2,2]))
        roll1 = -np.arctan2(Rmat[2,1],Rmat[2,2])
        roll2 = np.arctan2(Rmat[2,1],Rmat[2,2])

        self.roll = roll1*180/np.pi
        self.pitch = pitch1*180/np.pi
        self.yaw = yaw1*180/np.pi

        return self.roll, self.pitch, self.yaw

if __name__ == '__main__':
    a = LaserPosEstimator()
    
