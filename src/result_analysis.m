%% =========================================================
%  Population Spatial Classification (S1–S10)
%  3 classifiers: Skewed, Quadrant, 8-Class
%  Displacement-based affine per subject
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

%% ===================== POPULATION CONFUSION MATRICES =====================
C_skew = confusionmat(all_true_card, all_pred_card, 'Order', 1:4);
C_quad = confusionmat(all_true_quad, all_pred_quad, 'Order', 1:4);
C_8    = confusionmat(all_true_8,    all_pred_8,    'Order', 1:8);

% acc_section_skew = diag(C_skew) ./ sum(C_skew,2);
% acc_section_quad = diag(C_quad) ./ sum(C_quad,2);
% acc_section_8    = diag(C_8)    ./ sum(C_8,2);
%% ===================== CONFUSION MATRICES (COUNTS) =====================

% ---- Cardinal (Skewed) ----
figure;
confusionchart(all_true_card, all_pred_card);

title('Population Confusion Matrix — Cardinal (Counts)');
xlabel('Predicted Class');
ylabel('True Class');


% ---- Quadrant ----
figure;
confusionchart(all_true_quad, all_pred_quad);

title('Population Confusion Matrix — 4 Quadrant (Counts)');
xlabel('Predicted Quadrant');
ylabel('True Quadrant');


% ---- 8-Class ----
figure;
confusionchart(all_true_8, all_pred_8);

title('Population Confusion Matrix — 8 Class (Counts)');
xlabel('Predicted Class');
ylabel('True Class');

%% ===================== POPULATION SPATIAL PLOTS =====================
GAx_endpoints = all_dx_true;
GAy_endpoints = all_dy_true;

%% ===================== ENDPOINT-BASED SECTOR ASSIGNMENT =====================

N = length(GAx_endpoints);

sector_card = zeros(N,1);
sector_quad = zeros(N,1);
sector_8    = zeros(N,1);

for i = 1:N
    [card_i, quad_i, class8_i] = classify_num(GAx_endpoints(i), GAy_endpoints(i));
    sector_card(i) = card_i;     % 4-class skewed
    sector_quad(i) = quad_i;     % 4-class quadrant
    sector_8(i)    = class8_i;   % 8-class
end

%% ===================== ENDPOINT-BASED ACCURACY =====================

% ---- Skewed (Cardinal) ----
acc_section_skew_endpoint = zeros(4,1);
for k = 1:4
    idx = (sector_card == k);
    acc_section_skew_endpoint(k) = sum(all_pred_card(idx) == all_true_card(idx)) / sum(idx);
end

% ---- Quadrant ----
acc_section_quad_endpoint = zeros(4,1);
for k = 1:4
    idx = (sector_quad == k);
    acc_section_quad_endpoint(k) = sum(all_pred_quad(idx) == all_true_quad(idx)) / sum(idx);
end

% ---- 8-Class ----
idx_correct_8 = (all_pred_8 == all_true_8);
acc_section_8_endpoint = zeros(8,1);
n_section_8_endpoint   = zeros(8,1);

for k = 1:8
    idx = (sector_8 == k);
    n_section_8_endpoint(k) = sum(idx);
    if n_section_8_endpoint(k) > 0
        acc_section_8_endpoint(k) = sum(idx_correct_8(idx)) / n_section_8_endpoint(k);
    else
        acc_section_8_endpoint(k) = NaN;
    end
end

lim = max(max(abs(GAx_endpoints)), max(abs(GAy_endpoints)));
x_vals = linspace(-lim, lim, 500);

%% ===================== CARDINAL (SKEWED) =====================
figure; hold on;

idx_correct = (all_pred_card == all_true_card);
scatter(GAx_endpoints(idx_correct), GAy_endpoints(idx_correct), 10, 'g', 'filled');
scatter(GAx_endpoints(~idx_correct), GAy_endpoints(~idx_correct), 10, 'r', 'filled');

% Boundaries y = x, y = -x
plot(x_vals,  x_vals, 'k--', 'LineWidth', 1.5);
plot(x_vals, -x_vals, 'k--', 'LineWidth', 1.5);

axis equal; grid on;
xlim([-lim lim]); ylim([-lim lim]);
xlabel('Horizontal displacement (deg)');
ylabel('Vertical displacement (deg)');
title('Population Spatial Plot — Cardinal (Skewed)');
legend('Correct','Incorrect');

% Sectional accuracy labels
text( lim*0.6, 0, sprintf('SQ1 %.1f%%', acc_section_skew_endpoint(1)*100))
text( 0, lim*0.7, sprintf('SQ2 %.1f%%', acc_section_skew_endpoint(2)*100), 'HorizontalAlignment','center')
text(-lim*0.7, 0, sprintf('SQ3 %.1f%%', acc_section_skew_endpoint(3)*100))
text( 0,-lim*0.7, sprintf('SQ4 %.1f%%', acc_section_skew_endpoint(4)*100), 'HorizontalAlignment','center')


%% ===================== QUADRANT =====================
figure; hold on;

idx_correct = (all_pred_quad == all_true_quad);
scatter(GAx_endpoints(idx_correct), GAy_endpoints(idx_correct), 10, 'g', 'filled');
scatter(GAx_endpoints(~idx_correct), GAy_endpoints(~idx_correct), 10, 'r', 'filled');

xline(0,'k--','LineWidth',1.5);
yline(0,'k--','LineWidth',1.5);

axis equal; grid on;
xlim([-lim lim]); ylim([-lim lim]);
xlabel('Horizontal displacement (deg)');
ylabel('Vertical displacement (deg)');
title('Population Spatial Plot — 4 Quadrant');
legend('Correct','Incorrect');

% Quadrant accuracy labels
text( lim*0.5,  lim*0.5, sprintf('Q1 %.1f%%', acc_section_quad_endpoint(1)*100))
text(-lim*0.5,  lim*0.5, sprintf('Q2 %.1f%%', acc_section_quad_endpoint(2)*100))
text(-lim*0.5, -lim*0.5, sprintf('Q3 %.1f%%', acc_section_quad_endpoint(3)*100))
text( lim*0.5, -lim*0.5, sprintf('Q4 %.1f%%', acc_section_quad_endpoint(4)*100))


%% ===================== 8-CLASS =====================
figure; hold on;

idx_correct = (all_pred_8 == all_true_8);
scatter(GAx_endpoints(idx_correct), GAy_endpoints(idx_correct), 10, 'g', 'filled');
scatter(GAx_endpoints(~idx_correct), GAy_endpoints(~idx_correct), 10, 'r', 'filled');

m1 = tand(22.5);
m2 = tand(67.5);

plot(x_vals,  m1*x_vals, 'k--', 'LineWidth', 1.5)
plot(x_vals, -m1*x_vals, 'k--', 'LineWidth', 1.5)
plot(x_vals,  m2*x_vals, 'k--', 'LineWidth', 1.5)
plot(x_vals, -m2*x_vals, 'k--', 'LineWidth', 1.5)

axis equal; grid on;
xlim([-lim lim]); ylim([-lim lim]);
xlabel('Horizontal displacement (deg)');
ylabel('Vertical displacement (deg)');
title('Population Spatial Plot — 8 Class');
legend('Correct','Incorrect');

% 8-class accuracy labels
text( lim*0.7,  0, sprintf('S1 %.1f%%', acc_section_8_endpoint(1)*100))
text( lim*0.5,  lim*0.5, sprintf('S2 %.1f%%', acc_section_8_endpoint(2)*100))
text( 0,        lim*0.7, sprintf('S3 %.1f%%', acc_section_8_endpoint(3)*100), 'HorizontalAlignment','center')
text(-lim*0.5,  lim*0.5, sprintf('S4 %.1f%%', acc_section_8_endpoint(4)*100))
text(-lim*0.7,  0, sprintf('S5 %.1f%%', acc_section_8_endpoint(5)*100))
text(-lim*0.5, -lim*0.5, sprintf('S6 %.1f%%', acc_section_8_endpoint(6)*100))
text( 0,       -lim*0.7, sprintf('S7 %.1f%%', acc_section_8_endpoint(7)*100), 'HorizontalAlignment','center')
text( lim*0.5, -lim*0.5, sprintf('S8 %.1f%%', acc_section_8_endpoint(8)*100))

%% ---- Count small-displacement endpoints (< 1 deg from origin) ----
endpoint_dist = sqrt(GAx_endpoints.^2 + GAy_endpoints.^2);
num_small = sum(endpoint_dist < 1);
pct_small = num_small / length(endpoint_dist) * 100;

fprintf('Number of endpoints with displacement < 1 deg: %d (%.2f%%)\n', ...
        num_small, pct_small);
% for S4: Number of endpoints with displacement < 1 deg: 96 (48.00%)
%%% ===== Identify small-displacement endpoints (< 1 deg) =====

% Compute Euclidean distance from origin
endpoint_dist = sqrt(GAx_endpoints.^2 + GAy_endpoints.^2);

% Logical index of small endpoints
idx_small = endpoint_dist < 1;

% Count
num_small = sum(idx_small);
pct_small = num_small / length(endpoint_dist) * 100;

fprintf('\nSmall-displacement endpoints (<1 deg): %d (%.2f%%)\n', ...
        num_small, pct_small);

%% ===== Determine which sectors these small endpoints fall into =====
% Preallocate
small_card = zeros(num_small,1);
small_quad = zeros(num_small,1);
small_8    = zeros(num_small,1);

% Extract the small endpoints
dx_small = GAx_endpoints(idx_small);
dy_small = GAy_endpoints(idx_small);

% Classify each small endpoint
for i = 1:num_small
    [c, q, e] = classify_num(dx_small(i), dy_small(i));
    small_card(i) = c;
    small_quad(i) = q;
    small_8(i)    = e;
end

%% ===== Tabulate results =====

fprintf('\nSmall-endpoint distribution (Cardinal):\n');
tabulate(small_card)

fprintf('\nSmall-endpoint distribution (Quadrant):\n');
tabulate(small_quad)

fprintf('\nSmall-endpoint distribution (8-Class):\n');
tabulate(small_8)

%% ============================================================
%   SPATIAL PLOTS WITH SMALL ENDPOINTS REMOVED (< 1 deg)
% ============================================================

fprintf('\n=== Generating spatial plots with small endpoints removed (<1 deg) ===\n');

% Compute displacement magnitude
endpoint_dist = sqrt(GAx_endpoints.^2 + GAy_endpoints.^2);

% Keep only endpoints >= 1 deg
idx_keep = endpoint_dist >= 1;

dx_f = GAx_endpoints(idx_keep);
dy_f = GAy_endpoints(idx_keep);

true_card_f = all_true_card(idx_keep);
pred_card_f = all_pred_card(idx_keep);

true_quad_f = all_true_quad(idx_keep);
pred_quad_f = all_pred_quad(idx_keep);

true_8_f = all_true_8(idx_keep);
pred_8_f = all_pred_8(idx_keep);

fprintf('Removed %d small endpoints. Remaining: %d\n', ...
        sum(~idx_keep), sum(idx_keep));

%% ============================================================
%   Helper: Compute accuracy per sector
% ============================================================

compute_acc = @(trueL, predL, nSec) arrayfun(@(s) ...
    sum(predL(trueL==s) == s) / sum(trueL==s) * 100, 1:nSec);

%% ============================================================
%   1. CARDINAL (SKEWED)
% ============================================================

card_acc = compute_acc(true_card_f, pred_card_f, 4);

figure; hold on;
idx_correct = (pred_card_f == true_card_f);

scatter(dx_f(idx_correct), dy_f(idx_correct), 10, 'g', 'filled');
scatter(dx_f(~idx_correct), dy_f(~idx_correct), 10, 'r', 'filled');

% Boundaries
lim = max(max(abs(dx_f)), max(abs(dy_f)));
x_vals = linspace(-lim, lim, 500);
xlim([-lim lim]); ylim([-lim lim]);
plot(x_vals,  x_vals, 'k--', 'LineWidth', 1.5);
plot(x_vals, -x_vals, 'k--', 'LineWidth', 1.5);

axis equal; grid on;
xlabel('Horizontal displacement (deg)');
ylabel('Vertical displacement (deg)');
title('Cardinal Spatial Plot (Small Endpoints Removed)');
legend('Correct','Incorrect');

% Accuracy labels
text( lim*0.6, 0, sprintf('SQ1 %.1f%%', card_acc(1)));
text( 0, lim*0.7, sprintf('SQ2 %.1f%%', card_acc(2)), 'HorizontalAlignment','center');
text(-lim*0.7, 0, sprintf('SQ3 %.1f%%', card_acc(3)));
text( 0,-lim*0.7, sprintf('SQ4 %.1f%%', card_acc(4)), 'HorizontalAlignment','center');

hold off;

%% ============================================================
%   2. QUADRANT
% ============================================================

quad_acc = compute_acc(true_quad_f, pred_quad_f, 4);

figure; hold on;
idx_correct = (pred_quad_f == true_quad_f);

scatter(dx_f(idx_correct), dy_f(idx_correct), 10, 'g', 'filled');
scatter(dx_f(~idx_correct), dy_f(~idx_correct), 10, 'r', 'filled');

xline(0,'k--','LineWidth',1.5);
yline(0,'k--','LineWidth',1.5);
xlim([-lim lim]); ylim([-lim lim]);
axis equal; grid on;
xlabel('Horizontal displacement (deg)');
ylabel('Vertical displacement (deg)');
title('Quadrant Spatial Plot (Small Endpoints Removed)');
legend('Correct','Incorrect');

% Accuracy labels
text( lim*0.5,  lim*0.5, sprintf('Q1 %.1f%%', quad_acc(1)));
text(-lim*0.5,  lim*0.5, sprintf('Q2 %.1f%%', quad_acc(2)));
text(-lim*0.5, -lim*0.5, sprintf('Q3 %.1f%%', quad_acc(3)));
text( lim*0.5, -lim*0.5, sprintf('Q4 %.1f%%', quad_acc(4)));

hold off;

%% ============================================================
%   3. 8-CLASS
% ============================================================

eight_acc = compute_acc(true_8_f, pred_8_f, 8);

figure; hold on;
idx_correct = (pred_8_f == true_8_f);

scatter(dx_f(idx_correct), dy_f(idx_correct), 10, 'g', 'filled');
scatter(dx_f(~idx_correct), dy_f(~idx_correct), 10, 'r', 'filled');

% 8-class boundaries
m1 = tand(22.5);
m2 = tand(67.5);

plot(x_vals,  m1*x_vals, 'k--', 'LineWidth', 1.5);
plot(x_vals, -m1*x_vals, 'k--', 'LineWidth', 1.5);
plot(x_vals,  m2*x_vals, 'k--', 'LineWidth', 1.5);
plot(x_vals, -m2*x_vals, 'k--', 'LineWidth', 1.5);
xlim([-lim lim]); ylim([-lim lim]);
axis equal; grid on;
xlabel('Horizontal displacement (deg)');
ylabel('Vertical displacement (deg)');
title('8-Class Spatial Plot (Small Endpoints Removed)');
legend('Correct','Incorrect');

% Accuracy labels
% r = lim * 0.7;
% for s = 1:8
%     ang = (s-0.5)*45;
%     text(r*cosd(ang), r*sind(ang), sprintf('S%d %.1f%%', s, eight_acc(s)));
% end
text( lim*0.7,  0, sprintf('S1 %.1f%%', eight_acc(1)))
text( lim*0.5,  lim*0.5, sprintf('S2 %.1f%%', eight_acc(2)))
text( 0,        lim*0.7, sprintf('S3 %.1f%%', eight_acc(3)), 'HorizontalAlignment','center')
text(-lim*0.6,  lim*0.5, sprintf('S4 %.1f%%', eight_acc(4)))
text(-lim*0.7,  0, sprintf('S5 %.1f%%', eight_acc(5)))
text(-lim*0.6, -lim*0.5, sprintf('S6 %.1f%%', eight_acc(6)))
text( 0,       -lim*0.7, sprintf('S7 %.1f%%', eight_acc(7)), 'HorizontalAlignment','center')
text( lim*0.5, -lim*0.5, sprintf('S8 %.1f%%', eight_acc(8)))


hold off;

%% ============================================================
% CONFUSION MATRICES (Low-Amplitude Trials Removed)
% ============================================================

fprintf('\n=== Confusion matrices with low-amplitude trials removed (<1 deg) ===\n');

% Filtered labels already computed earlier:
% true_card_f, pred_card_f, true_quad_f, pred_quad_f, true_8_f, pred_8_f

% ---- Cardinal ----
figure;
confusionchart(true_card_f, pred_card_f);
title('Cardinal Confusion Matrix (Low-Amplitude Trials Removed)');
xlabel('Predicted Class');
ylabel('True Class');

% ---- Quadrant ----
figure;
confusionchart(true_quad_f, pred_quad_f);
title('Quadrant Confusion Matrix (Low-Amplitude Trials Removed)');
xlabel('Predicted Quadrant');
ylabel('True Quadrant');

% ---- 8-Class ----
figure;
confusionchart(true_8_f, pred_8_f);
title('8-Class Confusion Matrix (Low-Amplitude Trials Removed)');
xlabel('Predicted Class');
ylabel('True Class');
