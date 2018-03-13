function F = FLIST4(X)
global AB BC AC angle1 angle2  angle3 

F = [(2*X(1)*X(2)*cosd(angle1)-X(1)^2 - X(2)^2 + AB^2);
     (2*X(1)*X(3)*cosd(angle2)-X(1)^2 - X(3)^2 + AC^2);
     (2*X(2)*X(3)*cosd(angle3)-X(2)^2 - X(3)^2 + BC^2)];

end