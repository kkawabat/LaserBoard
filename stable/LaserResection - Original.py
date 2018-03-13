from __future__ import division 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
import time
from numpy.linalg import inv
from numpy.linalg import norm


##----- METHODS ------

#return a rotational matrix given an array of axis and angle desired
def getRotMatrix(orient_array, angle_array):
        #define an identity matrix
        RotMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        #for each orientation and angle in the input arrays set rotation matrix to that rotation
        for o, a in zip(orient_array, angle_array):
                if o == 'y':
                        RotMatrix = np.dot(RotMatrix, np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]))
                elif o == 'x':
                        RotMatrix = np.dot(RotMatrix, np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]))
                else:
                        RotMatrix = np.dot(RotMatrix, np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]))
        return RotMatrix

# Generates Test Points 
def laserf(X): 
    f = np.zeros(3)
    f[0] = np.cos(np.radians(angle2)) - (VATest[0]*X[0] + VATest[1]*X[1] + VATest[2]*X[2])/np.sqrt(np.dot(np.transpose(X),X))  
    f[1] = np.cos(np.radians(angle3)) - (VBTest[0]*X[0] + VBTest[1]*X[1] + VBTest[2]*X[2])/np.sqrt(np.dot(np.transpose(X),X)) 
    f[2] = 0
    return f

# Solves the Resection problem for lengths
def resectf(X):
    f = np.zeros(3)
    f[0] = (2*X[0]*X[1]*np.cos(np.radians(angle1))-X[0]*X[0] - X[1]*X[1] + AB*AB)
    f[1] = (2*X[0]*X[2]*np.cos(np.radians(angle2))-X[0]*X[0] - X[2]*X[2] + AC*AC)
    f[2] = (2*X[1]*X[2]*np.cos(np.radians(angle3))-X[1]*X[1] - X[2]*X[2] + BC*BC)
    return f

# Solves the Directions Vectors of each laser
def vectf(X):
##    print('RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
##    print(str(AP))
    f = np.zeros(9)
    f[0] = AP*(X[0]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - BP*(X[3]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5] ))) + (a[0] - b[0])
    f[1] = AP*(X[1]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - BP*(X[4]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5]))) + (a[1] - b[1])
    f[2] = AP*(X[2]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - BP*(X[5]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5]))) 
    f[3] = AP*(X[0]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - CP*(X[6]/(np.sqrt(X[6]*X[6] + X[7]*X[7]  + X[8]*X[8]))) + (a[0] - c[0])
    f[4] = AP*(X[1]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - CP*(X[7]/(np.sqrt(X[6]*X[6] + X[7]*X[7]  + X[8]*X[8]))) + (a[1] - c[1])
    f[5] = AP*(X[2]/(np.sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]))) - CP*(X[8]/(np.sqrt(X[6]*X[6] + X[7]*X[7]  + X[8]*X[8]))) 
    f[6] = CP*(X[6]/(np.sqrt(X[6]*X[6] + X[7]*X[7] + X[8]*X[8]))) - BP*(X[3]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5]))) + (c[0] - b[0])
    f[7] = CP*(X[7]/(np.sqrt(X[6]*X[6] + X[7]*X[7] + X[8]*X[8]))) - BP*(X[4]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5]))) + (c[1] - b[1])
    f[8] = CP*(X[8]/(np.sqrt(X[6]*X[6] + X[7]*X[7] + X[8]*X[8]))) - BP*(X[5]/(np.sqrt(X[3]*X[3] + X[4]*X[4]  + X[5]*X[5]))) 
    return f

##-------- Test Points ------------------

# angles between lasers
global angle1
global angle2
global angle3
angle1 = 16; 
angle2 = 15; 
angle3 = 13;


# Set Position Test Point
XTest = np.array([[1], [4], [10]])
# Set Orientation (all zero is pointing straigt to wall) 3-2-1 
yaw = 125;
pitch = 40;
roll =15;

zTest = np.array([[0],[0],[-1]]); # point towards wall
R = getRotMatrix(['z', 'y', 'x'], [np.radians(yaw), np.radians(pitch), np.radians(roll)])

VATest = np.dot(R,zTest);
VBTest = np.dot(np.dot(R,getRotMatrix(['y'],[np.radians(angle1)])),zTest)
VCTest0 = np.dot(np.dot(np.dot(R,getRotMatrix(['z'],[np.radians(-30*np.cos(np.radians(angle2)))])),getRotMatrix(['x'],[np.radians(-angle2)])),VATest)
tic = time.clock()
VCTest0 = np.array([[.03],[-.06],[-.06]])
VCTest  = fsolve(laserf,VCTest0)
VCTest2  = broyden1(laserf,VCTest0)
toc = time.clock()
print(toc-tic)  
VCTest = VCTest
print(VCTest)

#forcing VCTest to be what matlab found it to be
VCTest = np.array([[.3992],[-.6751],[-.6656]])
VCTest = VCTest/norm(VCTest)
print('forcing VCtest to match matlab...')
print(VCTest)

Aa = np.array([[1, 0, VATest[0]],
               [0, 1, VATest[1]],
               [0, 0, VATest[2]]]);

Ab = np.array([[1, 0, VBTest[0]],
               [0, 1, VBTest[1]],
               [0, 0, VBTest[2]]]);

Ac = np.array([[1, 0, VCTest[0]],
               [0, 1, VCTest[1]],
               [0, 0, VCTest[2]]]);

bb = XTest;

aTest = np.dot(inv(Aa),bb);
bTest = np.dot(inv(Ab),bb);
cTest = np.dot(inv(Ac),bb);

thetaABTest = np.arccos((np.dot(np.transpose(VATest),VBTest))/(norm(VATest)*norm(VBTest)))*180/np.pi
thetaACTest = np.arccos((np.dot(np.transpose(VATest),VCTest))/(norm(VATest)*norm(VCTest)))*180/np.pi
thetaCBTest = np.arccos((np.dot(np.transpose(VCTest),VBTest))/(norm(VCTest)*norm(VBTest)))*180/np.pi

print('test angles: ab, ac, cb')
print(thetaABTest)
print(thetaACTest)
print(thetaCBTest)

#obtain points of apparent triangle 
a = np.array([aTest[0],aTest[1]]);
b = np.array([bTest[0],bTest[1]]);
c = np.array([cTest[0],cTest[1]]);

##--------Solution-------------------------------

#obtain lengths of apparent triangle sides
AB = norm(b-a)
BC = norm(c-b)
AC = norm(c-a)

# solve resection problem for position
X0 = np.array([[norm(VATest*aTest[2])*.3],[norm(VBTest*bTest[2])*.5],[1.6*norm(VCTest*cTest[2])]])
print('guess for resect lengths')
print(X0)
tic1 = time.clock()
X = fsolve(resectf,X0)
toc1 = time.clock()
print('solution for resect lengths')
print(X)


AP = X[0]
BP = X[1]
CP = X[2]

#solving resection problem for orientation (direction vectors)
V0 = -np.array([VATest[0]*.4,VATest[1]*.9,VATest[2],VBTest[0]*.95,VBTest[1]*1.3,VBTest[2],VCTest[0]*1.3,VCTest[1]*1.2,VCTest[2]*.8]);
print('guess for resect direction vector')
print(V0)
tic2 = time.clock()
V = fsolve(vectf,V0)
toc2 = time.clock()
V[0:3] = V[0:3]/norm(V[0:3])
V[3:6] = V[3:6]/norm(V[3:6])
V[6:9] = V[6:9]/norm(V[6:9])

print('solution for resect direction vector')
print(V)

ABx = b[0] - a[0];
ABy = b[1] - a[1];
ACx = c[0] - a[0];
ACy = c[1] - a[1];
BCx = c[0] - b[0];
BCy = c[1] - b[1];

XYZa = np.array([a[0],a[1],0]) + np.dot(V[0:3],AP)
XYZb = np.array([b[0],b[1],0]) + np.dot(V[3:6],BP)
XYZc = np.array([c[0],c[1],0]) + np.dot(V[6:9],CP)


VA = np.dot(np.transpose(V[0:3]),AP)
VB = np.dot(np.transpose(V[3:6]),BP)
VC = np.dot(np.transpose(V[6:9]),CP)

print('Percent Change in Laser Lengths')
ChangePercentInAPLength = ((AP-norm(VA))/AP)*100
ChangePercentInBPLength = ((BP-norm(VB))/BP)*100
ChangePercentInCPLength = ((CP-norm(VC))/CP)*100
print(ChangePercentInAPLength)
print(ChangePercentInBPLength)
print(ChangePercentInCPLength)

print('Estimated  Angles Beteen Lasers After Resection Solved')
thetaAB = np.arccos((np.dot(np.transpose(VA),VB))/(norm(VA)*norm(VB)))
thetaAC = np.arccos((np.dot(np.transpose(VA),VC))/(norm(VA)*norm(VC)))
thetaCB = np.arccos((np.dot(np.transpose(VC),VB))/(norm(VC)*norm(VB)))
print(thetaAB*180/np.pi)
print(thetaAC*180/np.pi)
print(thetaCB*180/np.pi)

print('Time Elapsed for finding Position')
print(toc1 - tic1)

print('Time Elapsed for finding Direction Vectors')
print(toc2 - tic2)

print('Estimated Position')
print(XYZa)
