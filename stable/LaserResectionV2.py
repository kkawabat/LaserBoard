from __future__ import division 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import broyden1
import time
from numpy.linalg import inv
from numpy.linalg import norm

# ------------ GLOBAL VARIABLES ----------
global angle1
global angle2
global angle3
global AB
global BC
global AC
global AP
global BP
global CP

## ------------- END OF GLOBAL VARIABLES -----------

## -------------- SETUP -------------------
# angles between lasers
angle1 = 15; 
angle2 = 14; 
angle3 = 13;
## ------------- END OF SETUP -------------

## ------------- TEST POINTS --------------
angle1 = 15; 
angle2 = 14; 
angle3 = 13;

# actual position
XTest = np.transpose(np.array([[0, 0, 2], [ .07, .05, 2.11], [ .15, .12, 2.25]]))
print('XTest:')
print(XTest)
# Laser Points
aTestData = np.transpose(np.array([[-.1750, -.1756, 0], [ -.1816, -.0465, 0], [ -.2079, .0973, 0]]))
bTestData = np.transpose(np.array([[-.7301, -.1799, 0], [ -.7729, -.1008, 0], [ -.8439, -.0131, 0]]))
cTestData = np.transpose(np.array([[-.4836, -.5969, 0], [ -.4672, -.5129, 0], [ -.4698, -.4218, 0]]))
# 'Intial' Measurements/Solutions for intializing the algorithm
Xi = XTest[:,0]
print('Xi:')
print(np.transpose(Xi))
## ------------- END OF TEST POINTS ------------



## ----------- METHODS --------------

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

# The main function, solves for position given three lasers
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
    tocV = time.clock()

    # Ensure Directions are Normalized 
    V[0:3] = V[0:3]/norm(V[0:3])
    V[3:6] = V[3:6]/norm(V[3:6])
    V[6:9] = V[6:9]/norm(V[6:9])

    print('V:')
    print(V)

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
## -------------- END OF METHODS --------------


FirstPointCheck = 0
count = 0
while count < 3:
    count = count + 1
    print('--------- count:' +  str(count))
    if FirstPointCheck == 0:
        SolutionData = getPos(aTestData[:,count-1],bTestData[:,count-1],cTestData[:,count-1],Xi)
        X = SolutionData[0:3]
        print('Actual X:')
        print(XTest[:,count-1])
        FirstPointCheck =1

    else:
        SolutionData = getPos(aTestData[:,count-1],bTestData[:,count-1],cTestData[:,count-1],X)
        print('Actual X:')
        print(XTest[:,count-1])
        


        
        
        

    
    


