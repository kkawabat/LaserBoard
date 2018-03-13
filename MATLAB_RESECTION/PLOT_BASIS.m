function [  ] = PLOT_BASIS(P,B,Axis1,Axis2,Axis3,LineColorType,Size)
%PLOT_BASIS plots a 3d basis in the graphing frame.
%   P = Origin Position Expressed in graphing frame
%   B = 3x3 matrix of column vectors
%   Axis1 = name displayed for axis 1
%   Axis2 = name displayed for axis 2
%   Axis3 = name displayed for axis 3
%   LineColorType = plotting line color and line type
%   Size = Line thickness (recommend 14)
%
%   Ex:  PLOT_BASIS(P,I,'X','Y','Z','r--',14)
%
O1 =[P(1),P(1),P(1)];
O2 = [P(2),P(2),P(2)];
O3 = [P(3),P(3),P(3)];
X = [B(1,:)];
Y = [B(2,:)];
Z = [B(3,:)];

%Linespec.quiver3 = r;
quiver3(O1,O2,O3,X,Y,Z,LineColorType);
text(B(1,1)+P(1),B(2,1)+P(2),B(3,1)+P(3),Axis1,'FontSize',Size);
text(B(1,2)+P(1),B(2,2)+P(2),B(3,2)+P(3),Axis2,'FontSize',Size);
text(B(1,3)+P(1),B(2,3)+P(2),B(3,3)+P(3),Axis3,'FontSize',Size);

%whitebg([.2 .2 .2]);
whitebg([1 1 1]);
axis vis3d
end
