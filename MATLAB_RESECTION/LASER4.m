function V = LASER4(X)
global VATest VBTest angle2 angle3

 V = [cosd(angle2) - (VATest(1)*X(1) + VATest(2)*X(2) + VATest(3)*X(3))/sqrt(X(1)^2 + X(2)^2 + X(3)^2);
     cosd(angle3) - (VBTest(1)*X(1) + VBTest(2)*X(2) + VBTest(3)*X(3))/sqrt(X(1)^2 + X(2)^2 + X(3)^2);
     0];
end