function [card, quad, dir8] = classify_num(dx, dy)

% Cardinal
% 1 = Right, 2 = Up, 3 = Left, 4 = Down

% Quadrant
% 1 = UR, 2 = UL, 3 = DL, 4 = DR

% 8-class
% 1 = Right, 2 = Up-Right, 3 = Up, 4 = UL, 5 = Left, 6 = LD, 7 = D, 8 = DR


% Cardinal (dominant axis)
if abs(dx) > abs(dy)
    card = 1;
    if dx < 0
        card = 3;
    end
else
    card = 2;
    if dy < 0
        card = 4;
    end
end

% Quadrant
if dx > 0 && dy > 0
    quad = 1;
elseif dx < 0 && dy > 0
    quad = 2;
elseif dx < 0 && dy < 0
    quad = 3;
else
    quad = 4;
end

% 8-class
theta = atan2d(dy, dx);

if theta >= 67.5 && theta < 112.5
    dir8 = 3;
elseif theta >= 22.5 && theta < 67.5
    dir8 = 2;
elseif theta >= -22.5 && theta < 22.5
    dir8 = 1;
elseif theta >= -67.5 && theta < -22.5
    dir8 = 8;
elseif theta >= -112.5 && theta < -67.5
    dir8 = 7;
elseif theta >= -157.5 && theta < -112.5
    dir8 = 6;
elseif theta >= 157.5 || theta < -157.5
    dir8 = 5;
else
    dir8 = 4;
end
end
