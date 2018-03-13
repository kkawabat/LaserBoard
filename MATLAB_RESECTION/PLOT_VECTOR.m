function [  ] = PLOT_VECTOR(P,V,Name,LineColorType,Size)
%PLOT_BASIS plots a 3d vector in the graphing frame.
%
%   P = Origin Position in Graphing Frame
%   V = 3x1 column vector
%   Name = name displayed for vector
%   LineColorType = plotting line color and line type
%   Size = Line thickness (recommend 14)
%
%   Ex:  PLOT_VECTOR(V,'Position Vector','r-',14)
%
X = [V(1)];
Y = [V(2)];
Z = [V(3)];
O1 =P(1);
O2 =P(2);
O3 =P(3);
%Linespec.quiver3 = r;
quiver3(O1,O2,O3,X,Y,Z,LineColorType,'Autoscale','off');
text(V(1)+O1,V(2)+O2,V(3)+O3,Name,'FontSize',Size);
%whitebg([.2 .2 .2]);
whitebg([1 1 1]);
axis vis3d
end

