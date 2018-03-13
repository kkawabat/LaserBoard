function V = CONFIG4(X)
global LA LB angle2 angle3

 V = [cosd(angle2) - (LA(1)*X(1) + LA(2)*X(2) + LA(3)*X(3))/sqrt(X(1)^2 + X(2)^2 + X(3)^2);
     cosd(angle3) - (LB(1)*X(1) + LB(2)*X(2) + LB(3)*X(3))/sqrt(X(1)^2 + X(2)^2 + X(3)^2);
     0];
end