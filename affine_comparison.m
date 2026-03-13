clc;
clear;
close all;

Fs = 256;
num_subjects = 10;
num_trials = 200;
samples_per_trial = 4 * Fs;
fix_win = round(0.14 * Fs);

% Preallocate accuracy storage
% --- Standard affine (A + c)
acc_card_std = zeros(num_subjects,1);
acc_quad_std = zeros(num_subjects,1);
acc_8_std    = zeros(num_subjects,1);

% --- Displacement affine 
acc_card_disp = zeros(num_subjects,1);
acc_quad_disp = zeros(num_subjects,1);
acc_8_disp    = zeros(num_subjects,1);


%% ===================== LOOP OVER SUBJECTS =====================
for subj = 1:num_subjects

    fprintf('\nProcessing Subject %d...\n', subj);

     % -------- Load subject data for 10 subjects --------
    % The dataset used in this project is provided by the University of Malta.
    % It is not included in this repository.
    %
    % Please download the dataset from the official source listed in the README.
    % After downloading, update the folder path below to match the location of
    % the dataset on your local machine.
    
    folder = sprintf('PATH_TO_DATASET/S%d', subj);
    
    load(fullfile(folder,'EOG.mat'));
    load(fullfile(folder,'ControlSignal.mat'));
    load(fullfile(folder,'Target_GA_stream.mat'));

    % -------- Construct HEOG & VEOG --------
    HEOG = EOG(1,:) - EOG(2,:);
    VEOG = EOG(4,:) - EOG(3,:);

    %% ===================== SUBJECT-LEVEL AFFINE CALIBRATION =====================
    dEOG_h = [];
    dEOG_v = [];
    dX = [];
    dY = [];

    for tr = 1:num_trials
        t0 = (tr-1)*samples_per_trial + 1;

        % Fixation before saccade
        fix1_end = t0 + Fs - 1;
        fix1_idx = fix1_end - fix_win + 1 : fix1_end;

        % Fixation after saccade
        fix2_end = t0 + 2*Fs - 1;
        fix2_idx = fix2_end - fix_win + 1 : fix2_end;

        % EOG displacement
        dEOG_h(end+1) = mean(HEOG(fix2_idx)) - mean(HEOG(fix1_idx));
        dEOG_v(end+1) = mean(VEOG(fix2_idx)) - mean(VEOG(fix1_idx));

        % Gaze displacement (degrees)
        dX(end+1) = mean(Target_GA_stream(1,fix2_idx)) ...
                  - mean(Target_GA_stream(1,fix1_idx));
        dY(end+1) = mean(Target_GA_stream(2,fix2_idx)) ...
                  - mean(Target_GA_stream(2,fix1_idx));
    end

    % Solve affine (displacement-based)
    M = [dX(:), dY(:)];
    theta_h = M \ dEOG_h(:);
    theta_v = M \ dEOG_v(:);

    A_disp = [theta_h(1), theta_h(2);
         theta_v(1), theta_v(2)];
    % ===================== STANDARD AFFINE CALIBRATION =====================
    EOG_h_cal = [];
    EOG_v_cal = [];
    X_cal = [];
    Y_cal = [];

    for tr = 1:num_trials
        t0 = (tr-1)*samples_per_trial + 1;

        % Fixation before saccade
        fix1_end = t0 + Fs - 1;
        fix1_idx = fix1_end - fix_win + 1 : fix1_end;

        % Fixation after saccade
        fix2_end = t0 + 2*Fs - 1;
        fix2_idx = fix2_end - fix_win + 1 : fix2_end;

        % Collect absolute points
        EOG_h_cal(end+1) = mean(HEOG(fix1_idx));
        EOG_v_cal(end+1) = mean(VEOG(fix1_idx));
        X_cal(end+1) = mean(Target_GA_stream(1,fix1_idx));
        Y_cal(end+1) = mean(Target_GA_stream(2,fix1_idx));

        EOG_h_cal(end+1) = mean(HEOG(fix2_idx));
        EOG_v_cal(end+1) = mean(VEOG(fix2_idx));
        X_cal(end+1) = mean(Target_GA_stream(1,fix2_idx));
        Y_cal(end+1) = mean(Target_GA_stream(2,fix2_idx));
    end

    M = [X_cal(:), Y_cal(:), ones(length(X_cal),1)];
    theta_h = M \ EOG_h_cal(:);
    theta_v = M \ EOG_v_cal(:);

    A_std = [theta_h(1), theta_h(2);
        theta_v(1), theta_v(2)];
    c_std = [theta_h(3); theta_v(3)];

    %% ===================== EVALUATION =====================
    
    pred_card = strings(num_trials,1);
    pred_quad = strings(num_trials,1);
    pred_8    = strings(num_trials,1);

    true_card = strings(num_trials,1);
    true_quad = strings(num_trials,1);
    true_8    = strings(num_trials,1);

    for tr = 1:num_trials
        t0 = (tr-1)*samples_per_trial + 1;
        sac_idx = t0 : t0 + Fs - 1;

        h = HEOG(sac_idx);
        v = VEOG(sac_idx);

        GAx = Target_GA_stream(1,sac_idx);
        GAy = Target_GA_stream(2,sac_idx);

        % Reconstruct trajectory
        % ---- Standard affine reconstruction
        traj_std = zeros(length(h),2);

        for i = 1:length(h)
            % Standard affine
            traj_std(i,:)  = (A_std \ ([h(i); v(i)] - c_std))';

        end

        % Displacements
        dx_std  = traj_std(end,1)  - traj_std(1,1);
        dy_std  = traj_std(end,2)  - traj_std(1,2);

        dh = h(end) - h(1); 
        dv = v(end) - v(1); 
        dGA = A_disp \ [dh; dv]; 
        % A_delta is displacement-calibrated matrix 
        dx_disp = dGA(1); 
        dy_disp = dGA(2);

        dx_true = GAx(end) - GAx(1);
        dy_true = GAy(end) - GAy(1);

        % Classification
        [pred_card_std(tr), pred_quad_std(tr), pred_8_std(tr)] = classify_all(dx_std, dy_std);
        [pred_card_disp(tr), pred_quad_disp(tr), pred_8_disp(tr)] = classify_all(dx_disp, dy_disp);
        [true_card(tr), true_quad(tr), true_8(tr)] = classify_all(dx_true, dy_true);
    end

    %% ===================== ACCURACY =====================
    acc_card_std(subj)  = mean(pred_card_std'  == true_card);
    acc_quad_std(subj)  = mean(pred_quad_std'  == true_quad);
    acc_8_std(subj)     = mean(pred_8_std'     == true_8);

    acc_card_disp(subj) = mean(pred_card_disp' == true_card);
    acc_quad_disp(subj) = mean(pred_quad_disp' == true_quad);
    acc_8_disp(subj)    = mean(pred_8_disp'    == true_8);

end

%% ===================== SUMMARY PLOT =====================
subjects = 1:num_subjects;

mean_std  = (acc_card_std  + acc_quad_std  + acc_8_std)  / 3;
mean_disp = (acc_card_disp + acc_quad_disp + acc_8_disp) / 3;

figure;
plot(subjects, mean_std*100,  '-o', 'LineWidth', 2); hold on;
plot(subjects, mean_disp*100, '-s', 'LineWidth', 2);

xlabel('Subject');
ylabel('Mean classification accuracy (%)');
title('Standard affine vs displacement affine calibration');
legend('Standard affine (A + c)', 'Displacement affine', 'Location','best');
grid on;

figure(2);
% plot(subjects, acc_card_std*100, '-s', 'LineWidth', 2);hold on 
% plot(subjects, acc_quad_std*100, '-s', 'LineWidth', 2);
% plot(subjects, acc_8_std*100, '-s', 'LineWidth', 2);
plot(subjects, acc_card_disp*100, '-s', 'LineWidth', 2);hold on
plot(subjects, acc_quad_disp*100, '-s', 'LineWidth', 2);
plot(subjects, acc_8_disp*100, '-s', 'LineWidth', 2);
xlabel('Subject');
ylabel('Classification accuracy (%)');
title('Standard affine vs displacement affine calibration');
% legend('Std skewed', 'Std 4-quadrant', 'Std 8-class','Disp skewed', 'Disp 4-quadrant', 'Disp 8-class');
legend('Disp skewed', 'Disp 4-quadrant', 'Disp 8-class');
grid on;
