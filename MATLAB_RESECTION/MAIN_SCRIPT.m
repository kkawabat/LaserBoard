clear all
clc
close all

% How to use: 
%    *To change position of laser origin, edit the XTest array = [x,y,z]
%    *To change orientation of laser, edit yaw pitch roll values. Note there
%    are someinteresting values found.
%    *To change the angles between the lasers, edit angle1,angle2,angle3
global AB BC AC angle1 angle2 angle3 AP BP CP a b c VATest VBTest LA LB LC

%Set angles between lasers
angle1 = 15; 
angle2 = 15; 
angle3 = 15; 

%% Test points

% Set Position Test Point
XTest = [1; 1.5; 3];

% Set Orientation (all zero is pointing straigt to wall) 3-2-1 
yaw = 160;
pitch = -37;
roll =12;

%This one gave me trouble.... I found that when yaw approaches 180 degrees,
%the solutions get worse and worse. try to keep yaw less than 90
% yaw = 175;
% pitch = -10;
% roll =-15;
% or
% yaw = 185;
% pitch = -10;
% roll =-15;

% Compare the next two ypr values, the first is poor and the second is
% great, though only roll changes by 1 deg. I think its a numeric issue.
% yaw = -90;
% pitch = -10;
% roll =34;
% vs....
% yaw = -90;
% pitch = -10;
% roll =35;

zTest = [0;0;-1]; % point towards wall
R = ROT(3,yaw*pi/180)*ROT(2,pitch*pi/180)*ROT(1,roll*pi/180) %Get rotation matrix

VATest = R*zTest;
VBTest = R*ROT(2,angle1*pi/180)*zTest;
VCTest0 = R*ROT(3,-30*cosd(angle2)*pi/180)*ROT(1,-angle2*pi/180)*VATest;
[VCTest,fval,exitflag,output]  = fsolve(@LASER4,VCTest0*.9)
VCTest = VCTest/norm(VCTest)

Aa = [1 0 VATest(1);
      0 1 VATest(2);
      0 0 VATest(3)];

Ab = [1 0 VBTest(1);
      0 1 VBTest(2);
      0 0 VBTest(3)];

Ac = [1 0 VCTest(1);
      0 1 VCTest(2);
      0 0 VCTest(3)];

  
bb = XTest;

aTest = inv(Aa)*bb;
bTest = inv(Ab)*bb;
cTest = inv(Ac)*bb;

thetaABTest = acosd((VATest'*VBTest)/(norm(VATest)*norm(VBTest)))
thetaACTest = acosd((VATest'*VCTest)/(norm(VATest)*norm(VCTest)))
thetaCBTest = acosd((VCTest'*VBTest)/(norm(VCTest)*norm(VBTest)))


%% The solution
% vectorpoints
a = aTest(1:2,1);
b = bTest(1:2,1);
c = cTest(1:2,1);

AB = norm(b-a)
BC = norm(c-b)
AC = norm(c-a)

APg = 10;
BPg = 10;
CPg = 10;
X0 = [APg,BPg,CPg]
X0 = [norm(VATest*aTest(3))*.9,norm(VBTest*bTest(3))*.95,norm(VCTest*cTest(3))]

tic
[X,fval,exitflag,output]  = fsolve(@FLIST4,X0*.9)
toc

AP = X(1)
BP = X(2)
CP = X(3)

% Different intial guesses
V0 = [0,0,1,0,0,1,0,0,1];
V0 = [VATest(1)*.9,VATest(2)*.95,VATest(3),VBTest(1)*.95,VBTest(2)*.9,VBTest(3),VCTest(1),VCTest(2),VCTest(3)];
V0 = [VATest(1)*.9,VATest(2)*.95,VATest(3),VBTest(1)*.95,VBTest(2)*.9,VBTest(3),VCTest(1),VCTest(2),VCTest(3)];

tic
[V,fval,exitflag,output]  = fsolve(@VLIST4,V0*.9)
toc

V(1:3) = V(1:3)/norm(V(1:3))
V(4:6) = V(4:6)/norm(V(4:6))
V(7:9) = V(7:9)/norm(V(7:9))

ABx = b(1) - a(1);
ABy = b(2) - a(2);
ACx = c(1) - a(1);
ACy = c(2) - a(2);
BCx = c(1) - b(1);
BCy = c(2) - b(2);
 
%Estimated Positions from Solving Resection Problem And Direction Vectors
% Note: V(1:3) goes from a to laser source, and like wise for b and c.
XYZa = [a(1);a(2);0] + V(1:3)'*AP
XYZb = [b(1);b(2);0] + V(4:6)'*BP
XYZc = [c(1);c(2);0] + V(7:9)'*CP


% ISSUE NEEDS TO BE ADDRESSED HERE---------------------------------------
% ensure we are on correct side of board
% WHY DOES THIS WORK?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if XYZa(3) < 0 
    XYZa(3) = -XYZa(3);
    V(3) = -V(3);
end
if XYZb(3) < 0 
    XYZb(3) = -XYZb(3);
    V(6) = -V(6);
end
if XYZc(3) < 0 
    XYZc(3) = -XYZc(3);
    V(9) = -V(9);
end

%  
figure(1)
hold on
plot3(a(1),a(2),0,'o','MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.3,0.3,0.3])
plot3(b(1),b(2),0,'o','MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.6,0.6,0.6])
plot3(c(1),c(2),0,'o','MarkerEdgeColor','r',...
    'MarkerFaceColor',[0.9,0.9,0.9])
% Position points
plot3(XTest(1),XTest(2),XTest(3),'rx')
plot3((XYZa(1)+XYZb(1)+XYZc(1))/3,(XYZa(2)+XYZb(2)+XYZc(2))/3,(XYZa(3)+XYZb(3)+XYZc(3))/3,'c^')
legend('laser point a','laser point b','laser point c','Actual Position','Solved Position')
fill3([-3,-3,3,3],[-3,3,3,-3],[0,0,0,0],'c')
PLOT_VECTOR([0;0;0],[1;0;0],'I','w',14)
PLOT_VECTOR([0;0;0],[0;1;0],'J','w',14)
PLOT_VECTOR([0;0;0],[0;0;1],'K','w',14)

%test points
PLOT_VECTOR(XTest,[a(1),a(2),0]'-XTest,'a','r',14)
PLOT_VECTOR(XTest,[b(1),b(2),0]'-XTest,'b','r',14)
PLOT_VECTOR(XTest,[c(1),c(2),0]'-XTest,'c','r',14)

%solution
% Note: These are vectors from laser source to points a,b,c
VA = -V(1:3)'*AP ;
VB = -V(4:6)'*BP ;
VC = -V(7:9)'*CP ;

% BAD WAY OF PLOTTING RESULTS! : starts at solution and then cheats by adding
% a vector formed by subtracting known points on canvas by solution.
quiver3(XYZa(1),XYZa(2),XYZa(3),a(1)-XYZa(1),a(2)-XYZa(2),0-XYZa(3),'Autoscale','off','color','c')
quiver3(XYZb(1),XYZb(2),XYZb(3),b(1)-XYZb(1),b(2)-XYZb(2),0-XYZb(3),'Autoscale','off','color','c')
quiver3(XYZc(1),XYZc(2),XYZc(3),c(1)-XYZc(1),c(2)-XYZc(2),0-XYZc(3),'Autoscale','off','color','c')

% CORRECT WAY OF PLOTTING RESULTS! : starts at solution then adds the
% direction vector solution found.
PLOT_VECTOR(XYZa,VA,'a-sol','c',14)
PLOT_VECTOR(XYZb,VB,'b-sol','c',14)
PLOT_VECTOR(XYZc,VC,'c-sol','c',14)

whitebg([.3 .3 .3]);
grid on
axis on
alpha(0.2)
axis equal


ChangePercentInAPLength = ((AP-norm(VA))/AP)*100
ChangePercentInBPLength = ((BP-norm(VB))/BP)*100
ChangePercentInCPLength = ((CP-norm(VC))/CP)*100


thetaAB = acosd((VA'*VB)/(norm(VA)*norm(VB)))
thetaAC = acosd((VA'*VC)/(norm(VA)*norm(VC)))
thetaCB = acosd((VC'*VB)/(norm(VC)*norm(VB)))


% thetaABTest = acosd((VATest'*VBTest)/(norm(VATest)*norm(VBTest)))
% thetaACTest = acosd((VATest'*VCTest)/(norm(VATest)*norm(VCTest)))
% thetaCBTest = acosd((VCTest'*VBTest)/(norm(VCTest)*norm(VBTest)))
title('Laser Simulation and Solution')

hold off

figure(2) % plot direction vectors (not position)
hold on
% Given Orientation
PLOT_VECTOR([0;0;1],VATest,'A','r',14)
PLOT_VECTOR([0;0;1],VBTest,'B','r',14)
PLOT_VECTOR([0;0;1],VCTest,'C','r',14)
% Solution Orientation
PLOT_VECTOR([0;0;1],VA*(1/AP),'As','c',14)
PLOT_VECTOR([0;0;1],VB*(1/BP),'Bs','c',14)
PLOT_VECTOR([0;0;1],VC*(1/CP),'Cs','c',14)
% Plot axis
PLOT_VECTOR([0;0;0],[1;0;0],'I','w',14)
PLOT_VECTOR([0;0;0],[0;1;0],'J','w',14)
PLOT_VECTOR([0;0;0],[0;0;1],'K','w',14)
whitebg([.3 .3 .3]);
grid on
axis on
alpha(0.2)
axis equal
title('Compare Laser Direction Vectors to those of the Found Solution')

%% Obtain Angles
LA = [0;0;-1]; % point towards wall
LB = ROT(2,angle1*pi/180)*LA;
LC0 = ROT(3,-30*cosd(angle2)*pi/180)*ROT(1,-angle2*pi/180)*LA;
[LC,fval,exitflag,output]  = fsolve(@CONFIG4,LC0)

% double check these are correct by measuring angles between
thetaLAB = acosd((LA'*LB)/(norm(LA)*norm(LB)))
thetaLAC = acosd((LA'*LC)/(norm(LA)*norm(LC)))
thetaLCB = acosd((LC'*LB)/(norm(LC)*norm(LB)))

figure(3) % plot lasers pointing at wall (not position)
hold on
% initial orientation (no ypr)
PLOT_VECTOR([0;0;1],LA,'Ao','g',14) 
PLOT_VECTOR([0;0;1],LB,'Bo','g',14)
PLOT_VECTOR([0;0;1],LC,'Co','g',14)
% Given Orientation
PLOT_VECTOR([0;0;1],VATest,'A','r',14)
PLOT_VECTOR([0;0;1],VBTest,'B','r',14)
PLOT_VECTOR([0;0;1],VCTest,'C','r',14)
% Solution Orientation
PLOT_VECTOR([0;0;1],VA*(1/AP),'As','c',14)
PLOT_VECTOR([0;0;1],VB*(1/BP),'Bs','c',14)
PLOT_VECTOR([0;0;1],VC*(1/CP),'Cs','c',14)
% Axis
PLOT_VECTOR([0;0;0],[1;0;0],'I','w',14)
PLOT_VECTOR([0;0;0],[0;1;0],'J','w',14)
PLOT_VECTOR([0;0;0],[0;0;1],'K','w',14)
whitebg([.3 .3 .3]);
grid on
axis on
alpha(0.2)
axis equal
title('Compare Laser Direction Vectors to those of the Found Solution VS. the original')


% solve rotation matrix : For each Direction Vector: x = R*x0. Doing this for each direction vector and rearanging we get the system:
%                         [X0]*[r] = [X] --> [r] = inv([X0]'*[X0])*[X0]'*[X]

A = [LA(1) LA(2) LA(3) 0     0     0     0     0     0;
     0     0     0     LA(1) LA(2) LA(3) 0     0     0;
     0     0     0     0     0     0     LA(1) LA(2) LA(3);
     LB(1) LB(2) LB(3) 0     0     0     0     0     0;
     0     0     0     LB(1) LB(2) LB(3) 0     0     0;
     0     0     0     0     0     0     LB(1) LB(2) LB(3);
     LC(1) LC(2) LC(3) 0     0     0     0     0     0;
     0     0     0     LC(1) LC(2) LC(3) 0     0     0;
     0     0     0     0     0     0     LC(1) LC(2) LC(3)];
     
DV = [VA*(1/AP);VB*(1/BP);VC*(1/CP)];
r = inv(A'*A)*A'*DV
0
Rmat = [r(1) r(2) r(3);
        r(4) r(5) r(6);
        r(7) r(8) r(9)]
    
R
    
yaw = atan2d(Rmat(2,1),Rmat(1,1))
pitch1 = atan2d(-Rmat(3,1),sqrt(Rmat(3,2)^2 + Rmat(3,3)^2))
pitch2 = atan2d(Rmat(3,1),sqrt(Rmat(3,2)^2 + Rmat(3,3)^2))
roll2 = atan2d(Rmat(3,2),Rmat(3,3))
roll1 = -atan2d(Rmat(3,2),Rmat(3,3))
R1 = ROT(3,yaw*pi/180)*ROT(2,pitch1*pi/180)*ROT(1,roll1*pi/180) %Get rotation matrix
R2 = ROT(3,yaw*pi/180)*ROT(2,pitch2*pi/180)*ROT(1,roll2*pi/180) %Get rotation matrix

Difference1 = dist2(R,R1); % measure difference from known R to calculated R1
Difference2 = dist2(R,R2); % measure differecnce from known R to calculated R2



R
syms psi phi theta
ROT(3,psi)*ROT(2,theta)*ROT(1,phi)






