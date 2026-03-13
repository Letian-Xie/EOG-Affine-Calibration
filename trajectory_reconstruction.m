clc
clear
close all
%% Load all the required files for Subject n (Sn)
% The dataset used in this project is provided by the University of Malta.
% It is not included in this repository.
%
% Please download the dataset from the official source listed in the README.
% After downloading, update the folder path below to match the location of
% the dataset on your local machine.

folder = sprintf('PATH_TO_DATASET/S%d', subj);
filename1 = 'EOG.mat';
filename2 = 'ControlSignal.mat';
filename3 = 'Target_GA_stream.mat';
load(fullfile(folder,filename1));
load(fullfile(folder,filename2));
load(fullfile(folder,filename3));
Fs = 256; % from data description document
%% Construct HEOG & VEOG
HEOG = EOG(1,:) - EOG(2,:);
VEOG = EOG(4,:) - EOG(3,:);

% ---- isolate ONE saccade (first trial, first saccade) ----
%% ---- Isolate first saccade of a chosen trial ----
trial_num = 18;   % <<< CHANGE THIS NUMBER to choose the trial (1–200)

samples_per_trial = 4 * Fs;

% Compute trial boundaries
t_start = (trial_num - 1) * samples_per_trial + 1;
t_end   = trial_num * samples_per_trial;

% Extract ControlSignal for this trial only
ctrl_trial = ControlSignal(t_start:t_end);

% Find all samples belonging to the FIRST saccade in this trial
idx_local = find(ctrl_trial == 1);

if isempty(idx_local)
    error('No saccade found in trial %d', trial_num);
end

% Find where the first saccade ends (gap > 1)
d_local = diff(idx_local);
break_idx = find(d_local > 1, 1, 'first');

if isempty(break_idx)
    idx_local_sacc = idx_local;
else
    idx_local_sacc = idx_local(1:break_idx);
end

% Convert local indices back to global indices
sacc_idx = idx_local_sacc + (t_start - 1);

% Extract signals for this saccade
h = HEOG(sacc_idx);
v = VEOG(sacc_idx);

GAx = Target_GA_stream(1, sacc_idx);
GAy = Target_GA_stream(2, sacc_idx);

true_dx = GAx(end) - GAx(1);
true_dy = GAy(end) - GAy(1);

[card_true, quad_true, dir8_true] = classify_all(true_dx, true_dy);
fprintf('\n=== TRUE SACCADE ===\n');  
fprintf('Quadrant: %s\n', quad_true);
fprintf('Cardinal: %s\n', card_true); 
fprintf('8-Class: %s\n', dir8_true);
%% Select quasi-fixation samples
% Parameters
fix_win = round(0.2 * Fs);   % 200 ms fixation window -> has 51 samples (where eye is relatively stable, average to reduce error in EMG and noise 

% Containers
% each entry is one calibration point, 2 calibration points per trial, at the start of each saccade 
EOG_h_cal = []; % EOG value at the calibration point 
EOG_v_cal = [];
X = []; % actual gaze angle at the calibration point
Y = [];

% Identify trial boundaries (4 seconds each)
samples_per_trial = 4 * Fs; % 4s * fs (unit: samples/s) -> gives #samples per trial
num_trials = 200; % from data description document

for tr = 1:num_trials
    % Trial start index
    t0 = (tr-1)*samples_per_trial + 1; % first sample of the current trial
    
    % ---- Fixation before first saccade (P2 -> P1) ----
    fix1_end = t0 + Fs - 1;           % index of sample at the end of 1st second
    fix1_idx = (fix1_end - fix_win + 1) : fix1_end; % the indices for all the samples inside the fix window
    
    % ---- Fixation before second saccade (P1 -> P2) ----
    fix2_end = t0 + 2*Fs - 1;         % end of 2nd second
    fix2_idx = fix2_end - fix_win + 1 : fix2_end;
    
    % Collect EOG (average during fixation)
    EOG_h_cal(end+1) = mean(HEOG(fix1_idx));
    EOG_v_cal(end+1) = mean(VEOG(fix1_idx));
    X(end+1) = mean(Target_GA_stream(1, fix1_idx));
    Y(end+1) = mean(Target_GA_stream(2, fix1_idx));
    
    EOG_h_cal(end+1) = mean(HEOG(fix2_idx));
    EOG_v_cal(end+1) = mean(VEOG(fix2_idx));
    X(end+1) = mean(Target_GA_stream(1, fix2_idx));
    Y(end+1) = mean(Target_GA_stream(2, fix2_idx));
end
%% Affine calibration (regression, find A and C)
M = [X(:), Y(:), ones(length(X),1)];

theta_h = M \ EOG_h_cal(:);
theta_v = M \ EOG_v_cal(:);

A = [theta_h(1), theta_h(2);
     theta_v(1), theta_v(2)];

c = [theta_h(3); theta_v(3)];
%% Trajectory reconstruction (inverse affine)
trajectory = zeros(length(h), 2);

for i = 1:length(h)
    E = [h(i); v(i)];
    trajectory(i,:) = (A \ (E - c))';
end

X_traj = trajectory(:,1);
Y_traj = trajectory(:,2);

% Displacement vector
dx = X_traj(end) - X_traj(1);
dy = Y_traj(end) - Y_traj(1);
%% ===================== PLOT RECONSTRUCTED VS TRUE TRAJECTORY =====================
figure;

plot(X_traj, Y_traj, '-o', 'LineWidth', 1.5); hold on;
plot(GAx, GAy, '-o', 'LineWidth', 2);

xlabel('X (deg)');
ylabel('Y (deg)');
title(sprintf('Trial %d: Reconstructed vs True Gaze Trajectory', trial_num));
legend('Reconstructed (EOG)', 'True Gaze (GA)', 'Location', 'best');
axis equal;
grid on;

hold off;


