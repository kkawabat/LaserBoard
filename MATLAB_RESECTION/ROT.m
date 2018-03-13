function [R] = ROT( AXIS,RAD)
% Inputs: AXIS = Axis of Frame I which to rotate about 
%         RAD = Angle in rad which to rotate (Right Hand Rule)
% Outputs: Rotation matrix which rotates a vector expressed in the current
%          frame
%
% Use:  There is a distinction between arbitrarily rotating a vector about 
%       a basis axis and rotating a basis frame about itself. The proof 
%       wont be discussed here but the rule of how to do each are given:
%
% "Rotating A Vector"
%
%       V'_I = R_I_II*V_I = ROT(AXIS,RAD)*V_I
%
%       Here a vector expressed in I (V_I) was rotated about the (AXIS)
%       axis of frame I by an angle (RAD), which formed a new vector still
%       expressed in I (V'_I).
%
% "Rotating A Basis (or frame)"
%
%       [II]_G = [I]_G*R_I_II = [I]_G*ROT(AXIS,ROT)
%
%       Here a basis [I]_G expressed in the graphing frame is rotated about
%       its own (AXIS) axis by an angle (RAD) which forms a new basis
%       [II]_G, which is still expressed in the graphing frame
%
% Graphics: Since both [I]_G and [II]_G are expressed in the graphin frame,
%           simply plotting them will yeild the correct result. If the
%           frames are not expressed in the graphing frame, then steps
%           should be done to first have them expressed in the graphing
%           frame.
if AXIS == 3
 R = [cos(RAD),-sin(RAD), 0; sin(RAD), cos(RAD) 0; 0 0 1];
end

if AXIS == 2
R = [cos(RAD), 0, sin(RAD); 0 1 0; - sin(RAD), 0 , cos(RAD)];
end

if AXIS == 1
R = [1 0 0; 0, cos(RAD) -sin(RAD); 0, sin(RAD) cos(RAD)];
end


end

