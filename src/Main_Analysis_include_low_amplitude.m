%% Reproduces the main analysis pipeline used in the report
%% =========================================================
%  Population Spatial Classification (S1–S10)
%  3 classifiers: Skewed, Quadrant, 8-Class
%  Displacement-based affine per subject
% This file contain confusion matrices and spatial plots for 
% both the low-amplitude filtered and unfiltered. 
% =========================================================

clc
clear
close all

%% ===================== PARAMETERS =====================
Fs = 256;
num_trials = 200;
num_subjects = 10;
samples_per_trial = 4 * Fs;
fix_win = round(0.14 * Fs);

%% ===================== STORAGE FOR POPULATION =====================
all_dx_true = [];
all_dy_true = [];

all_true_card = [];
all_pred_card = [];

all_true_quad = [];
all_pred_quad = [];

all_true_8 = [];
all_pred_8 = [];

%% ===================== LOOP OVER SUBJECTS =====================
for subj = 1:num_subjects

    fprintf('Processing Subject %d...\n', subj);

    % Load subject data
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

    HEOG = EOG(1,:) - EOG(2,:);
    VEOG = EOG(4,:) - EOG(3,:);

    %% ===================== DISPLACEMENT AFFINE CALIBRATION =====================
    dEOG_h = zeros(num_trials,1);
    dEOG_v = zeros(num_trials,1);
    dX     = zeros(num_trials,1);
    dY     = zeros(num_trials,1);

    for tr = 1:num_trials
        trial_idx = (tr-1)*samples_per_trial + 1 : tr*samples_per_trial;
        cs_trial  = ControlSignal(trial_idx);

        fix1 = trial_idx(cs_trial==1);
        fix2 = trial_idx(cs_trial==2);

        % Skip trial if segment too short
        if length(fix1) < fix_win || length(fix2) < fix_win
            continue
        end

        fix1 = fix1(end-fix_win+1:end);
        fix2 = fix2(end-fix_win+1:end);


        dEOG_h(tr) = mean(HEOG(fix2)) - mean(HEOG(fix1));
        dEOG_v(tr) = mean(VEOG(fix2)) - mean(VEOG(fix1));

        dX(tr) = mean(Target_GA_stream(1,fix2)) - mean(Target_GA_stream(1,fix1));
        dY(tr) = mean(Target_GA_stream(2,fix2)) - mean(Target_GA_stream(2,fix1));
    end

    % Solve displacement affine
    M = [dX(:), dY(:)];
    theta_h = M \ dEOG_h(:);
    theta_v = M \ dEOG_v(:);

    A = [theta_h(1), theta_h(2);
         theta_v(1), theta_v(2)];

    %% ===================== EVALUATION =====================
    dx_true_all = zeros(num_trials,1);
    dy_true_all = zeros(num_trials,1);

    true_card_num = zeros(num_trials,1);
    pred_card_num = zeros(num_trials,1);

    true_quad_num = zeros(num_trials,1);
    pred_quad_num = zeros(num_trials,1);

    true_8_num = zeros(num_trials,1);
    pred_8_num = zeros(num_trials,1);

    for tr = 1:num_trials
        t0 = (tr-1)*samples_per_trial + 1;
        sac_idx = t0 : t0 + Fs - 1;

        h = HEOG(sac_idx);
        v = VEOG(sac_idx);

        dh = h(end) - h(1);
        dv = v(end) - v(1);

        dGA = A \ [dh; dv];
        dx_pred = dGA(1);
        dy_pred = dGA(2);

        GAx = Target_GA_stream(1,sac_idx);
        GAy = Target_GA_stream(2,sac_idx);

        dx_true = GAx(end) - GAx(1);
        dy_true = GAy(end) - GAy(1);

        dx_true_all(tr) = dx_true;
        dy_true_all(tr) = dy_true;

        [pred_card_num(tr), pred_quad_num(tr), pred_8_num(tr)] = classify_num(dx_pred, dy_pred);
        [true_card_num(tr), true_quad_num(tr), true_8_num(tr)] = classify_num(dx_true, dy_true);
    end

    %% ===================== APPEND TO POPULATION =====================
    all_dx_true = [all_dx_true; dx_true_all];
    all_dy_true = [all_dy_true; dy_true_all];

    all_true_card = [all_true_card; true_card_num];
    all_pred_card = [all_pred_card; pred_card_num];

    all_true_quad = [all_true_quad; true_quad_num];
    all_pred_quad = [all_pred_quad; pred_quad_num];

    all_true_8 = [all_true_8; true_8_num];
    all_pred_8 = [all_pred_8; pred_8_num];

end
%% ===================== POPULATION CONFUSION MATRICES (PERCENT) =====================
C_skew = confusionmat(all_true_card, all_pred_card, 'Order', 1:4);
C_quad = confusionmat(all_true_quad, all_pred_quad, 'Order', 1:4);
C_8    = confusionmat(all_true_8,    all_pred_8,    'Order', 1:8);

acc_section_skew = diag(C_skew) ./ sum(C_skew,2);
acc_section_quad = diag(C_quad) ./ sum(C_quad,2);
acc_section_8    = diag(C_8)    ./ sum(C_8,2);

% Convert counts → row‑wise percentages
C_skew_pct = 100 * (C_skew ./ sum(C_skew,2));
C_quad_pct = 100 * (C_quad ./ sum(C_quad,2));
C_8_pct    = 100 * (C_8    ./ sum(C_8,2));



% Replace NaN (rows with zero samples) with 0
C_skew_pct(isnan(C_skew_pct)) = 0;
C_quad_pct(isnan(C_quad_pct)) = 0;
C_8_pct(isnan(C_8_pct))       = 0;

%% ---------- Cardinal (Skewed) ----------
figure;
imagesc(C_skew_pct);
axis equal tight;

title('Population Confusion Matrix — Cardinal (Skewed)');
xlabel('Predicted Class');
ylabel('True Class');

xticks(1:4); yticks(1:4);

% Add percentage labels
for i = 1:4
    for j = 1:4
        text(j, i, sprintf('%.1f%%', C_skew_pct(i,j)), ...
            'HorizontalAlignment','center', 'Color','k', 'FontSize',10);
    end
end


%% ---------- 4‑Quadrant ----------
figure;
imagesc(C_quad_pct);
axis equal tight;

title('Population Confusion Matrix — 4 Quadrant');
xlabel('Predicted Quadrant');
ylabel('True Quadrant');

xticks(1:4); yticks(1:4);

for i = 1:4
    for j = 1:4
        text(j, i, sprintf('%.1f%%', C_quad_pct(i,j)), ...
            'HorizontalAlignment','center', 'Color','k', 'FontSize',10);
    end
end


%% ---------- 8‑Class ----------
figure;
imagesc(C_8_pct);
axis equal tight;

title('Population Confusion Matrix — 8 Class');
xlabel('Predicted Class');
ylabel('True Class');

xticks(1:8); yticks(1:8);

for i = 1:8
    for j = 1:8
        text(j, i, sprintf('%.1f%%', C_8_pct(i,j)), ...
            'HorizontalAlignment','center', 'Color','k', 'FontSize',9);
    end
end
