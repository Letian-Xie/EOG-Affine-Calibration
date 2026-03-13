clc
clear
close all
%% Load all the required files for Subject n
% The dataset used in this project is provided by the University of Malta.
% It is not included in this repository.
%
% Please download the dataset from the official source listed in the README.
% After downloading, update the folder path below to match the location of
% the dataset on your local machine.
    
folder = sprintf('PATH_TO_DATASET');
filename1 = 'EOG.mat';
filename2 = 'ControlSignal.mat';
filename3 = 'Target_GA_stream.mat';
load(fullfile(folder,filename1));
load(fullfile(folder,filename2));
load(fullfile(folder,filename3));
Fs = 256; % from data description document
%% ===================== SETUP =====================
HEOG = EOG(1,:) - EOG(2,:);
VEOG = EOG(4,:) - EOG(3,:);

samples_per_trial = 4 * Fs;
num_trials = 200;
fix_win = round(0.14 * Fs); % get the fixation window, where gaze are relatively stable -> used as reference points in affine calibration

%% ===================== AFFINE CALIBRATION =====================
EOG_h_cal = [];
EOG_v_cal = [];
X = [];
Y = [];


for tr = 1:num_trials
    t0 = (tr-1)*samples_per_trial + 1; % t0: start of each trial

    % Fixation before saccade 1
    fix1_end = t0 + Fs - 1;
    fix1_idx = fix1_end - fix_win + 1 : fix1_end; % so have exact the no. of samples as in fix_win

    % Fixation before saccade 2
    fix2_end = t0 + 2*Fs - 1;
    fix2_idx = fix2_end - fix_win + 1 : fix2_end;

    EOG_h_cal(end+1) = mean(HEOG(fix1_idx));
    EOG_v_cal(end+1) = mean(VEOG(fix1_idx));
    X(end+1) = mean(Target_GA_stream(1,fix1_idx));
    Y(end+1) = mean(Target_GA_stream(2,fix1_idx));

    EOG_h_cal(end+1) = mean(HEOG(fix2_idx));
    EOG_v_cal(end+1) = mean(VEOG(fix2_idx));
    X(end+1) = mean(Target_GA_stream(1,fix2_idx));
    Y(end+1) = mean(Target_GA_stream(2,fix2_idx));
end

M = [X(:), Y(:), ones(length(X),1)];
theta_h = M \ EOG_h_cal(:);
theta_v = M \ EOG_v_cal(:);

A = [theta_h(1), theta_h(2);
     theta_v(1), theta_v(2)];
c = [theta_h(3); theta_v(3)];

%% ===================== EVALUATION =====================
pred_card = strings(num_trials,1);
pred_quad = strings(num_trials,1);
pred_8    = strings(num_trials,1);

true_card = strings(num_trials,1);
true_quad = strings(num_trials,1);
true_8    = strings(num_trials,1);

for tr = 1:num_trials
    t0 = (tr-1)*samples_per_trial + 1;

    % First saccade (0–1 s) 
    sac_idx = t0 : t0 + Fs - 1;

    h = HEOG(sac_idx);
    v = VEOG(sac_idx);

    GAx = Target_GA_stream(1,sac_idx);
    GAy = Target_GA_stream(2,sac_idx);

    % Affine inverse reconstruction (Reconstructed trajectory is in gaze ANGLE space (degrees))
    traj = zeros(length(h),2);
    for i = 1:length(h)
        traj(i,:) = (A \ ([h(i); v(i)] - c))';
    end
    
    % Angular displacement of saccade (degrees)
    dx_pred = traj(end,1) - traj(1,1);
    dy_pred = traj(end,2) - traj(1,2);
    % now added end point averagging
    % dx_pred = mean(traj(end-40:end,1)) - mean(traj(1:10,1));
    % dy_pred = mean(traj(end-40:end,2)) - mean(traj(1:10,2));

    dx_true = GAx(end) - GAx(1);
    dy_true = GAy(end) - GAy(1);

    % Same classifiers for both
    [pred_card(tr), pred_quad(tr), pred_8(tr)] = classify_num(dx_pred, dy_pred);
    [true_card(tr), true_quad(tr), true_8(tr)] = classify_num(dx_true, dy_true);
end

%% ===================== ACCURACY =====================
acc_card = mean(pred_card == true_card);
acc_quad = mean(pred_quad == true_quad);
acc_8    = mean(pred_8 == true_8);

fprintf('Cardinal accuracy: %.2f %%\n', acc_card*100);
fprintf('Quadrant accuracy: %.2f %%\n', acc_quad*100);
fprintf('8-class accuracy:  %.2f %%\n', acc_8*100);

%% ===================== PLOT SPATIAL PLOT =====================
% Matrix of the GA of the endpoints
% Indices of the 256th sample of each trial 
endpoint_idx = (0:num_trials-1) * samples_per_trial + Fs; 
% Extract GAx and GAy endpoints 
GAx_endpoints = Target_GA_stream(1, endpoint_idx); 
GAy_endpoints = Target_GA_stream(2, endpoint_idx);

xmax = max(abs(GAx_endpoints));
xmin = -xmax;
ymax = max(abs(GAy_endpoints));
x_vals = linspace(xmin, xmax, 500);
lim  = max(xmax, ymax);

y1 =  x_vals;    % y = x
y2 = -x_vals;    % y = -x

% spatial plot for skewed
figure(1); hold on
% 1. Correct points
idx1 = (pred_card == true_card); 
GAx_correct_card = GAx_endpoints(idx1); 
GAy_correct_card = GAy_endpoints(idx1);
scatter(GAx_correct_card,GAy_correct_card, 20, 'green','filled');
% 2. Incorrect points
idx2 = (pred_card ~= true_card); 
GAx_wrong_card = GAx_endpoints(idx2); 
GAy_wrong_card = GAy_endpoints(idx2);
scatter(GAx_wrong_card,GAy_wrong_card, 20, 'red','filled');
% 3. Boundaries
plot(x_vals, y1, 'k--', 'LineWidth', 1.5);
plot(x_vals, y2, 'k--', 'LineWidth', 1.5);
% 4. Labeling
axis equal;
grid on;
xlim([-lim lim]);
ylim([-lim lim]);
xlabel('Horizontal gaze angle (deg)');
ylabel('Vertical gaze angle (deg)');
title('Spatial distribution of classification accuracy: Skewed Quadrant');
legend('Correct', 'Incorrect');

% spatial plot for 4 quadrant
figure(2); hold on
% 1. Correct points
idx3 = (pred_quad == true_quad); 
GAx_correct_quad = GAx_endpoints(idx3); 
GAy_correct_quad = GAy_endpoints(idx3);
scatter(GAx_correct_quad,GAy_correct_quad, 20, 'green','filled');
% 2. Incorrect points
idx4 = (pred_quad ~= true_quad); 
GAx_wrong_quad = GAx_endpoints(idx4); 
GAy_wrong_quad = GAy_endpoints(idx4);
scatter(GAx_wrong_quad,GAy_wrong_quad, 20, 'red','filled');
% 3. Boundaries
xline(0, 'k--', 'LineWidth', 1.5); 
yline(0, 'k--', 'LineWidth', 1.5);
% 4. Labeling
axis equal;
grid on;
xlim([-lim lim]);
ylim([-lim lim]);
xlabel('Horizontal gaze angle (deg)');
ylabel('Vertical gaze angle (deg)');
title('Spatial distribution of classification accuracy: 4 Quadrant');
legend('Correct', 'Incorrect');

% spatial plot for 8-class
figure(3); hold on 
% 1. Correct points
idx5 = (pred_8 == true_8); 
GAx_correct_8 = GAx_endpoints(idx5); 
GAy_correct_8 = GAy_endpoints(idx5);
scatter(GAx_correct_8,GAy_correct_8, 20, 'green','filled');
% 2. Incorrect points
idx6 = (pred_8 ~= true_8); 
GAx_wrong_8 = GAx_endpoints(idx6); 
GAy_wrong_8 = GAy_endpoints(idx6);
scatter(GAx_wrong_8,GAy_wrong_8, 20, 'red','filled');
% 3. Boundaries
xline(0, 'k--', 'LineWidth', 1.5); 
yline(0, 'k--', 'LineWidth', 1.5);
plot(x_vals, y1, 'k--', 'LineWidth', 1.5);
plot(x_vals, y2, 'k--', 'LineWidth', 1.5);
% 4. Labeling
axis equal;
grid on;
xlim([-lim lim]);
ylim([-lim lim]);
xlabel('Horizontal gaze angle (deg)');
ylabel('Vertical gaze angle (deg)');
title('Spatial distribution of classification accuracy: 8 Class');
legend('Correct', 'Incorrect');
