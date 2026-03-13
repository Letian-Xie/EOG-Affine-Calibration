clc;
clear;
close all;

Fs = 256;
num_subjects = 10;
num_trials = 200;
samples_per_trial = 4 * Fs;
fix_win = round(0.14 * Fs);

% Preallocate accuracy storage
acc_card_all = zeros(num_subjects,1);
acc_quad_all = zeros(num_subjects,1);
acc_8_all    = zeros(num_subjects,1);

%% ===================== LOOP OVER SUBJECTS =====================
for subj = 1:num_subjects

    fprintf('\nProcessing Subject %d...\n', subj);

    % -------- Load subject data --------
    folder = sprintf('C:\\Users\\xiele\\Documents\\SCIE30001\\data set from Uni Malta\\S%d', subj);
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

    A = [theta_h(1), theta_h(2);
         theta_v(1), theta_v(2)];

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

        dh = h(end) - h(1); 
        dv = v(end) - v(1); 
        dGA = A \ [dh; dv]; 
        % A_delta is displacement-calibrated matrix 
        dx_pred = dGA(1); 
        dy_pred = dGA(2);
        GAx = Target_GA_stream(1,sac_idx);
        GAy = Target_GA_stream(2,sac_idx);

        % Reconstruct trajectory
        % traj = zeros(length(h),2);
        % for i = 1:length(h)
        %     traj(i,:) = (A \ [h(i); v(i)])';
        % end
        % 
        % % Displacement
        % dx_pred = traj(end,1) - traj(1,1);
        % dy_pred = traj(end,2) - traj(1,2);

        
        dx_true = GAx(end) - GAx(1);
        dy_true = GAy(end) - GAy(1);

        % Classification
        [pred_card(tr), pred_quad(tr), pred_8(tr)] = classify_all(dx_pred, dy_pred);
        [true_card(tr), true_quad(tr), true_8(tr)] = classify_all(dx_true, dy_true);
    end

    %% ===================== ACCURACY =====================
    acc_card_all(subj) = mean(pred_card == true_card);
    acc_quad_all(subj) = mean(pred_quad == true_quad);
    acc_8_all(subj)    = mean(pred_8 == true_8);

    fprintf('  Cardinal: %.2f %%\n', acc_card_all(subj)*100);
    fprintf('  Quadrant: %.2f %%\n', acc_quad_all(subj)*100);
    fprintf('  8-class:  %.2f %%\n', acc_8_all(subj)*100);
end

%% ===================== SUMMARY PLOT =====================
subjects = 1:num_subjects;
mean_acc = (acc_card_all + acc_quad_all + acc_8_all) / 3;

figure;
plot(subjects, acc_card_all*100, '-o', 'LineWidth', 2); hold on;
plot(subjects, acc_quad_all*100, '-s', 'LineWidth', 2);
plot(subjects, acc_8_all*100,    '-^', 'LineWidth', 2);
plot(subjects, mean_acc*100,     '--k', 'LineWidth', 2);

xlabel('Subject');
ylabel('Classification Accuracy (%)');
title('Classification accuracy across subjects');
legend('Cardinal','Quadrant','8-class','Mean (3 classifiers)', ...
       'Location','best');
grid on;
