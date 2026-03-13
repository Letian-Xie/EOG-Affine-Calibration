function [card, quad, dir8] = classify_all(dx, dy)

% Cardinal (dominant axis)
if abs(dx) > abs(dy)
    card = "Right";
    if dx < 0
        card = "Left";
    end
else
    card = "Up";
    if dy < 0
        card = "Down";
    end
end

% Quadrant
if dx > 0 && dy > 0
    quad = "Up-Right";
elseif dx < 0 && dy > 0
    quad = "Up-Left";
elseif dx < 0 && dy < 0
    quad = "Down-Left";
else
    quad = "Down-Right";
end

% 8-class
theta = atan2d(dy, dx);

if theta >= 67.5 && theta < 112.5
    dir8 = "Up";
elseif theta >= 22.5 && theta < 67.5
    dir8 = "Up-Right";
elseif theta >= -22.5 && theta < 22.5
    dir8 = "Right";
elseif theta >= -67.5 && theta < -22.5
    dir8 = "Down-Right";
elseif theta >= -112.5 && theta < -67.5
    dir8 = "Down";
elseif theta >= -157.5 && theta < -112.5
    dir8 = "Down-Left";
elseif theta >= 157.5 || theta < -157.5
    dir8 = "Left";
else
    dir8 = "Up-Left";
end
end
