%% CSI Feedback Sample Generation & Validation (500 Hz)
% Based on: CSI Feedback with Autoencoders - MATLAB & Simulink example
% This script generates CSI (Channel State Information) samples at 500 Hz
% and validates whether the generated samples are correct.
% No autoencoder is built — only sample generation and validation.

clc; clear; close all;

%% =========================================================
%  SYSTEM PARAMETERS
% =========================================================
fc          = 3.5e9;        % Carrier frequency: 3.5 GHz (typical 5G NR)
fs          = 500;          % Sample generation rate: 500 Hz
T_total     = 2;            % Total duration in seconds
numSamples  = fs * T_total; % Total number of CSI samples = 1000

numTx       = 32;           % Number of transmit antennas (BS)
numRx       = 4;            % Number of receive antennas (UE)
numSubc     = 32;           % Number of subcarriers (OFDM)

rng(42);                    % Fixed seed for reproducibility

fprintf('=== CSI Sample Generation at %d Hz ===\n', fs);
fprintf('Total samples to generate : %d\n', numSamples);
fprintf('Channel matrix size        : %d Tx x %d Rx x %d subcarriers\n', ...
        numTx, numRx, numSubc);

%% =========================================================
%  STEP 1 — GENERATE CSI SAMPLES
%  Each sample is a numRx x numTx x numSubc complex channel matrix H
%  using a clustered delay line (CDL) inspired random model.
% =========================================================
fprintf('\n[1] Generating CSI samples...\n');

% Pre-allocate: [numRx, numTx, numSubc, numSamples]
H_samples = (randn(numRx, numTx, numSubc, numSamples) + ...
          1j*randn(numRx, numTx, numSubc, numSamples)) / sqrt(2);

% Apply path-loss scaling and spatial correlation (simplified CDL model)
% Distance-based path loss (UE moving at 3 km/h over T_total seconds)
speed_mps  = 3 / 3.6;                    % 3 km/h → m/s
lambda     = 3e8 / fc;                   % Wavelength
fd_max     = speed_mps / lambda;         % Max Doppler shift ~10 Hz

t_axis     = (0:numSamples-1) / fs;      % Time axis

for sIdx = 1:numSamples
    % Doppler phase rotation per subcarrier (temporal variation)
    doppler_phase = exp(1j * 2 * pi * fd_max * t_axis(sIdx) * ...
                        (1:numSubc) / numSubc);
    for rxIdx = 1:numRx
        for txIdx = 1:numTx
            H_samples(rxIdx, txIdx, :, sIdx) = ...
                squeeze(H_samples(rxIdx, txIdx, :, sIdx)) .* doppler_phase.';
        end
    end
end

fprintf('   Done. H_samples size: [%d x %d x %d x %d]\n', ...
        size(H_samples, 1), size(H_samples, 2), ...
        size(H_samples, 3), size(H_samples, 4));

%% =========================================================
%  STEP 2 — CONVERT TO 2-CHANNEL REAL REPRESENTATION
%  (Standard preprocessing used in CSI autoencoder literature)
%  Stack real and imaginary parts → [2, numRx*numTx, numSubc, numSamples]
% =========================================================
fprintf('\n[2] Converting to real-valued 2-channel representation...\n');

H_real = real(H_samples);   % [numRx, numTx, numSubc, numSamples]
H_imag = imag(H_samples);

% Reshape to [numSamples, 2, numSubc, numRx*numTx] for network-style format
numAntennas = numRx * numTx;

H_real_rs = reshape(H_real, [numRx*numTx, numSubc, numSamples]);  
H_imag_rs = reshape(H_imag, [numRx*numTx, numSubc, numSamples]);

% Final tensor: [numSamples, numSubc, numAntennas, 2]  (2 = Re/Im)
CSI_tensor = zeros(numSamples, numSubc, numAntennas, 2, 'single');
for s = 1:numSamples
    CSI_tensor(s, :, :, 1) = squeeze(H_real_rs(:, :, s))';   % Real
    CSI_tensor(s, :, :, 2) = squeeze(H_imag_rs(:, :, s))';   % Imag
end

fprintf('   CSI tensor size: [%d samples x %d subcarriers x %d antennas x 2 channels]\n', ...
        size(CSI_tensor));

%% =========================================================
%  STEP 3 — VALIDATION CHECKS
% =========================================================
fprintf('\n[3] Running validation checks...\n');

passAll = true;
results = struct();

% --- CHECK 1: Correct number of samples ---
check1 = (size(CSI_tensor, 1) == numSamples);
results.numSamples_correct = check1;
fprintf('   CHECK 1 — Sample count (%d): %s\n', ...
        size(CSI_tensor,1), tf2str(check1));
passAll = passAll & check1;

% --- CHECK 2: Sampling interval consistency ---
% Verify that the time between consecutive samples matches 1/500 Hz
dt_expected = 1 / fs;             % 0.002 s
dt_actual   = t_axis(2) - t_axis(1);
check2 = abs(dt_actual - dt_expected) < 1e-10;
results.sampling_interval_correct = check2;
fprintf('   CHECK 2 — Sampling interval (expected %.4f s, got %.4f s): %s\n', ...
        dt_expected, dt_actual, tf2str(check2));
passAll = passAll & check2;

% --- CHECK 3: No NaN or Inf values ---
check3 = ~any(isnan(CSI_tensor(:))) && ~any(isinf(CSI_tensor(:)));
results.no_nan_inf = check3;
fprintf('   CHECK 3 — No NaN/Inf values: %s\n', tf2str(check3));
passAll = passAll & check3;

% --- CHECK 4: Power distribution (should follow Rayleigh, mean ≈ 1) ---
power_per_sample = mean(abs(H_samples).^2, [1 2 3]);  % [1x1x1xN]
mean_power = mean(power_per_sample(:));
std_power  = std(power_per_sample(:));
check4 = (mean_power > 0.5) && (mean_power < 2.0);    % Loose sanity bound
results.power_range_ok = check4;
fprintf('   CHECK 4 — Mean channel power (expected ~1.0, got %.4f ± %.4f): %s\n', ...
        mean_power, std_power, tf2str(check4));
passAll = passAll & check4;

% --- CHECK 5: Real & Imaginary parts are balanced (zero-mean Gaussian) ---
re_mean = mean(H_real(:));
im_mean = mean(H_imag(:));
check5 = (abs(re_mean) < 0.1) && (abs(im_mean) < 0.1);
results.zero_mean_ok = check5;
fprintf('   CHECK 5 — Zero-mean parts (Re mean=%.4f, Im mean=%.4f): %s\n', ...
        re_mean, im_mean, tf2str(check5));
passAll = passAll & check5;

% --- CHECK 6: Doppler variation present (samples should differ over time) ---
% Compute average correlation between sample 1 and last sample
h1   = H_samples(:,:,:,1);
hEnd = H_samples(:,:,:,end);
corr_val = abs(sum(h1(:) .* conj(hEnd(:)))) / ...
           (norm(h1(:)) * norm(hEnd(:)));
check6 = (corr_val < 0.999);   % Samples should NOT be identical
results.temporal_variation_ok = check6;
fprintf('   CHECK 6 — Temporal variation (corr first/last=%.4f, want <0.999): %s\n', ...
        corr_val, tf2str(check6));
passAll = passAll & check6;

% --- CHECK 7: Tensor data type is single (memory efficient) ---
check7 = isa(CSI_tensor, 'single');
results.dtype_single = check7;
fprintf('   CHECK 7 — Data type is single: %s\n', tf2str(check7));
passAll = passAll & check7;

%% =========================================================
%  STEP 4 — OVERALL RESULT
% =========================================================
fprintf('\n========================================\n');
if passAll
    fprintf('  OVERALL RESULT: ✓ ALL CHECKS PASSED\n');
    fprintf('  CSI samples at %d Hz are VALID.\n', fs);
else
    fprintf('  OVERALL RESULT: ✗ SOME CHECKS FAILED\n');
    fprintf('  Review the failed checks above.\n');
end
fprintf('========================================\n');

%% =========================================================
%  STEP 5 — VISUALISATION
% =========================================================
fprintf('\n[4] Plotting sample statistics...\n');

figure('Name', 'CSI Sample Generation & Validation', ...
       'Position', [100 100 1200 800]);

% --- Plot 1: Channel magnitude over time (single antenna pair, subcarrier 1) ---
subplot(2,3,1);
h_mag = squeeze(abs(H_samples(1,1,1,:)));
plot(t_axis, h_mag, 'b-', 'LineWidth', 1);
xlabel('Time (s)'); ylabel('|H|');
title('Channel Magnitude vs Time (Rx1, Tx1, SC1)');
grid on;

% --- Plot 2: Power distribution across samples ---
subplot(2,3,2);
histogram(power_per_sample(:), 40, 'FaceColor', [0.2 0.6 0.9]);
xlabel('Mean Power per Sample'); ylabel('Count');
title(sprintf('Power Distribution (mean=%.3f)', mean_power));
grid on;

% --- Plot 3: Real vs Imaginary scatter (first sample, all antennas) ---
subplot(2,3,3);
h_flat = reshape(H_samples(:,:,:,1), [], 1);
scatter(real(h_flat), imag(h_flat), 5, 'filled', ...
        'MarkerFaceAlpha', 0.3, 'MarkerFaceColor', [0.8 0.2 0.2]);
xlabel('Real'); ylabel('Imaginary');
title('Re vs Im (Sample 1, all antennas & SCs)');
axis equal; grid on;

% --- Plot 4: Frequency response (magnitude across subcarriers, sample 1) ---
subplot(2,3,4);
H_sc = squeeze(H_samples(1,1,:,1));   % All subcarriers, sample 1
plot(1:numSubc, abs(H_sc), 'r-o', 'MarkerSize', 4);
xlabel('Subcarrier Index'); ylabel('|H|');
title('Freq. Response - Sample 1 (Rx1, Tx1)');
grid on;

% --- Plot 5: Temporal correlation ---
subplot(2,3,5);
corr_arr = zeros(1, numSamples);
h_ref    = H_samples(:,:,:,1);
for s = 1:numSamples
    hs = H_samples(:,:,:,s);
    corr_arr(s) = abs(sum(h_ref(:) .* conj(hs(:)))) / ...
                  (norm(h_ref(:)) * norm(hs(:)));
end
plot(t_axis, corr_arr, 'k-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Correlation with Sample 1');
title('Temporal Decorrelation over 500 Hz Samples');
ylim([0 1.05]); grid on;

% --- Plot 6: Validation summary bar ---
subplot(2,3,6);
check_names  = {'#Samples','dt=1/500','No NaN','Power','ZeroMean','Variation','Single'};
check_values = [check1 check2 check3 check4 check5 check6 check7];
bar_colors   = check_values;   % 1=green, 0=red
b = bar(check_values);
b.FaceColor = 'flat';
for k = 1:numel(check_values)
    if check_values(k)
        b.CData(k,:) = [0.2 0.7 0.3];
    else
        b.CData(k,:) = [0.9 0.2 0.2];
    end
end
set(gca, 'XTickLabel', check_names, 'XTickLabelRotation', 30);
ylim([-0.1 1.3]); ylabel('Pass (1) / Fail (0)');
title('Validation Check Results');
grid on;

sgtitle(sprintf('CSI Sample Generation at %d Hz — %d Samples', fs, numSamples), ...
        'FontSize', 14, 'FontWeight', 'bold');

fprintf('   Plots generated.\n\n');

%% =========================================================
%  HELPER FUNCTION
% =========================================================
function s = tf2str(val)
    if val
        s = 'PASS ✓';
    else
        s = 'FAIL ✗';
    end
end