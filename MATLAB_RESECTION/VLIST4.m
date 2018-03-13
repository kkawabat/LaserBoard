function V = VLIST4(X)
global AP BP CP a b c

V = [AP*(X(1)/(sqrt(X(1)^2 + X(2)^2 + X(3)^2))) - BP*(X(4)/(sqrt(X(4)^2 + X(5)^2 + X(6)^2))) + (a(1) - b(1));
     AP*(X(2)/(sqrt(X(1)^2 + X(2)^2 + X(3)^2))) - BP*(X(5)/(sqrt(X(4)^2 + X(5)^2 + X(6)^2))) + (a(2) - b(2));
     AP*(X(3)/(sqrt(X(1)^2 + X(2)^2 + X(3)^2))) - BP*(X(6)/(sqrt(X(4)^2 + X(5)^2 + X(6)^2))) ;
     AP*(X(1)/(sqrt(X(1)^2 + X(2)^2 + X(3)^2))) - CP*(X(7)/(sqrt(X(7)^2 + X(8)^2 + X(9)^2))) + (a(1) - c(1));
     AP*(X(2)/(sqrt(X(1)^2 + X(2)^2 + X(3)^2))) - CP*(X(8)/(sqrt(X(7)^2 + X(8)^2 + X(9)^2))) + (a(2) - c(2));
     AP*(X(3)/(sqrt(X(1)^2 + X(2)^2 + X(3)^2))) - CP*(X(9)/(sqrt(X(7)^2 + X(8)^2 + X(9)^2))) ;
     CP*(X(7)/(sqrt(X(7)^2 + X(8)^2 + X(9)^2))) - BP*(X(4)/(sqrt(X(4)^2 + X(5)^2 + X(6)^2))) + (c(1) - b(1));
     CP*(X(8)/(sqrt(X(7)^2 + X(8)^2 + X(9)^2))) - BP*(X(5)/(sqrt(X(4)^2 + X(5)^2 + X(6)^2))) + (c(2) - b(2));
     CP*(X(9)/(sqrt(X(7)^2 + X(8)^2 + X(9)^2))) - BP*(X(6)/(sqrt(X(4)^2 + X(5)^2 + X(6)^2))) ];


end