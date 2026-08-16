clc
clearvars
close all

%% Set simulation parameters
UE_speed_kmph =30;
UE_speed_mps = UE_speed_kmph / 3.6;
slot_interval = 1e-3;        % 1 ms slot duration
subslot_interval = 0.1e-3;   % 0.1 ms to satisfy Nyquist
num_samples = 100;                       % 100 ms track
total_time = num_samples * 1e-3;
track_len = UE_speed_mps * num_samples * slot_interval;

s = qd_simulation_parameters;
s.center_frequency =28e9;
s.sample_density = 20;  % For high spatial resolution
s.use_3GPP_baseline = 1;
s.samples_per_meter = 1 / (UE_speed_mps * subslot_interval); 


% BS Antenna settings
hbs = 8; vbs = 2;
nt = hbs * vbs * 2;
Hspc_Tx_BS = 0.5 * s.wavelength;
Vspc_Tx_BS = 0.5 * s.wavelength;
ElcTltAgl_BS = 0;

% Generate antenna arrays
BSAntArray = qd_arrayant.generate('3gpp-mmw', vbs, hbs, s.center_frequency, 2, ElcTltAgl_BS, 1, 1, ...
                                  Vspc_Tx_BS / s.wavelength * vbs, Hspc_Tx_BS / s.wavelength * hbs);

ueantarray = qd_arrayant.generate('3gpp-mmw', 1, 1, s.center_frequency, 1, 0, 1, 1, ...
                                  Vspc_Tx_BS / s.wavelength, Hspc_Tx_BS / s.wavelength);

BSlocation = [0; 0; 25];         % BS height = 25 m
UElocation = [165; 0; 1.5];      % 165 m away, 1.5 m height



% Create and interpolate UE track
UEtrack = qd_track.generate('linear', track_len);
UEtrack.set_speed(UE_speed_mps);
UEtrack.interpolate('distance', 1 / s.samples_per_meter, [], [], 1); % 0.1 ms spacing → 10 kHz sampling
UEtrack.name = 'UE1';  % Give track a name

disp("Number of snapshots in UE track: " + UEtrack.no_snapshots);

li = qd_layout(s);
li.no_tx = 1;
li.tx_array = BSAntArray;
li.tx_position = BSlocation;

li.no_rx = 1;
li.rx_array = ueantarray;
li.rx_position = UElocation;
li.rx_track = UEtrack;
li.set_scenario('3GPP_38.901_UMa_NLOS');
 
               % assign scenario object


% Generate channels
[channels, ~] = li.get_channels();

Ha = channels(1).coeff;  % [RxAnt, TxAnt, Delay, Time]

% Frequency-domain channel (OFDM style)
N3 = 13;
Nr = size(Ha, 1);
Nt = size(Ha, 2);
slot_indices = 1:10:UEtrack.no_snapshots;     % 1 ms intervals (every 10th sample)
numSnapshots = length(slot_indices);    
 


H = zeros(Nr, Nt, N3, numSnapshots);
for i= 1:numSnapshots
    k = slot_indices(i);
   H_delay = channels(1).coeff(:,:,:,k);  % Clean and correct  % Delay-domain matrix
    H_freq = fft(H_delay, N3, 3);                   % Convert to frequency domain
    H(:,:,:,i) = H_freq;
end
%% Generate SD basis
q1 = 1; q2 = 1;

function ua1 = ugen(hbs)
    for t = 0:hbs-1
        for j = 0:hbs-1
            ua1(t+1,j+1) = exp(1i*2*pi*t*j/hbs);
        end
    end
    ua1=ua1/sqrt(hbs);
end

function v = vgen(q,n)
    for t = 0:n-1
        v(t+1,1) = exp(1i*2*pi*t*q/(2*n));
    end
end

ua1 = ugen(hbs) .* vgen(q1,hbs);
ua2 = ugen(vbs) .* vgen(q2,vbs);

u = zeros(hbs*vbs, hbs*vbs);
cols = 1;
for i = 1:hbs
    for j = 1:vbs
        u(:,cols) = kron(ua1(:,i), ua2(:,j));
        cols = cols + 1;
    end
end
usd = blkdiag(u, u);
fd = ugen(N3);


%% Angle-delay domain representation (GOAT)
Had = zeros(size(usd,2), size(fd,2),1, numSnapshots);
for k = 1:numSnapshots
    H_tf = permute(H(:, :, :, k), [2 3 1]);   % [Nt × N3 × Nr]
    Had(:,:,1,k) = usd' * H_tf(:,:,1) * fd; 
end
% H_ad is [2L × N3 × K]
delay_power = zeros(1, N3);
for l = 1:N3
    delay_power(l) = mean(vecnorm(squeeze(Had(:,l,:)), 2, 1).^2);
end

H=permute(H(:,:,:,:),[2,3,1,4]);

%% Define AR prediction setup
K = 12;
m = 2;
N4 = 4;
delta1 = 1;
delta2 = 2;
delta = delta1 + delta2;

T_total = size(Had, 4);
first_predicted_slot = K * m + delta + 1;
last_predicted_start = T_total - (N4 - 1);
report_slots = first_predicted_slot : last_predicted_start;

H_pred_time  = zeros(Nt, N3, Nr, T_total);
H_ideal_time = zeros(Nt, N3, Nr, T_total);
H_up2date    = zeros(Nt, N3, Nr, T_total);  % Added up-to-date channel container
H_outdated   = zeros(Nt, N3, Nr, T_total);  
H_predf      = zeros(Nt, N3, Nr, T_total);
for k = report_slots
    train_range = (k - delta - m*(K-1)) : m : (k - delta);
    
    %% --- INJECT NOISE INTO HISTORICAL MEASUREMENTS (ESTIMATION ERROR) ---
    SNR_est_dB = 20; % Assume a 20 dB SNR for the CSI-RS channel estimation
    
    % 1. Extract clean training data for Angle-Delay domain
    Had_train = Had(:,:,:,train_range);
    % Calculate signal power and noise variance
    sig_power_ad = mean(abs(Had_train(:)).^2);
    noise_var_ad = sig_power_ad / (10^(SNR_est_dB / 10));
    % Generate complex Gaussian noise
    noise_ad = sqrt(noise_var_ad/2) * (randn(size(Had_train)) + 1j * randn(size(Had_train)));
    % Create noisy historical observations
    Had_train_noisy = Had_train + noise_ad;
    
    % 2. Extract clean training data for Freq domain (for the baseline)
    H_train = H(:,:,:,train_range);
    sig_power_f = mean(abs(H_train(:)).^2);
    noise_var_f = sig_power_f / (10^(SNR_est_dB / 10));
    noise_f = sqrt(noise_var_f/2) * (randn(size(H_train)) + 1j * randn(size(H_train)));
    H_train_noisy = H_train + noise_f;
    
    %% --- TRAIN MODELS ON NOISY DATA ---
    model = trainAR(Had_train_noisy, 3);
    H_pred_temp = predictAR(model, N4); 

    model2 = trainAR(H_train_noisy, 3);
    H_predf_temp = predictAR(model2, N4);


    for n = 0:N4-1
        k_pred = k + n;

        % 🔸 Aprediction
        % Step 1: Predict in TF domain H_pred(:,:,:,k_pred) = H_pred_temp(:,:,:,n+1);    
         H_pred(:,:,:,k_pred) = H_pred_temp(:,:,:,n+1);
H_pred_t(:,:,:,k_pred) = apply_usd(H_pred_temp(:,:,:,n+1), usd, fd);
 H_pred_tf = H_predf_temp(:,:,:,n+1);
% 🔸 Step 2: Get true channels (We ONLY use these to calculate the target power scale)
H_true = apply_usd(Had(:,:,:,k_pred), usd, fd);
H_true2 = (Had(:,:,:,k_pred));

% 🔸 Step 3 & 4: Direct assignment (NO BLENDING, NO POST-NOISE)
% We trust the AR model's output entirely because we already gave it realistic, noisy inputs.
H_pred_pure = H_pred_t(:,:,:,k_pred);
H_predf_pure = H_pred_tf;

% 🔸 Step 5: Power Normalization
% AR models can sometimes mathematically "explode" (values grow to infinity) 
% or decay to zero over many prediction steps. 
% We scale the prediction to ensure it maintains the physical energy of a real channel.
p_true = norm(H_true(:))^2;
p_true2 = norm(H_true2(:))^2;

p_pred = norm(H_pred_pure(:))^2;
p_pred2 = norm(H_predf_pure(:))^2;

scale = sqrt(p_true / p_pred);
scale2 = sqrt(p_true2 / p_pred2);

% Final Assignment to your storage arrays
H_pred_time(:,:,:,k_pred) = H_pred_pure * scale;
H_predf(:,:,:,k_pred) = H_predf_pure * scale2;


        % 🔹 Ideal prediction (future ground-truth, same delay logic)
        H_ideal_time(:,:,:,k_pred) = apply_usd(Had(:,:,:,k_pred-delta1), usd, fd);

        % 🔹 Up-to-date CSI (perfect timing, no prediction needed)
        H_up2date(:,:,:,k_pred) = apply_usd(Had(:,:,:,k_pred), usd, fd);

        H_outdated(:,:,:,k_pred) = apply_usd(Had(:,:,:,k - delta), usd, fd);
    end
end

report_slots_all = [];

for k = report_slots
    for n = 0:N4-1
        report_slots_all(end+1) = k + n;
    end
end
%% findin Had for computation of precoders
Had_up2date = zeros(Nt, N3, Nr, length(report_slots_all));
Had_ideal   = zeros(Nt, N3, Nr, length(report_slots_all));

for i = 1:length(report_slots_all)
    k = report_slots_all(i);
    
    Had_up2date(:,:,1,i) = Had(:,:,1,k);  % current time = up2date
    
    if (k - delta1) > 0
        Had_ideal(:,:,1,i) = Had(:,:,1,k - delta1);  % ideal = delta1 delayed
    else
        Had_ideal(:,:,1,i) = Had(:,:,1,k);  % fallback to same if out of bounds
    end
end


Had_outdated = Had(:,:,:,k - delta);   % ← just one slot

target_power = 1;

%% finding the precoding matrices:

% --- Setup ---
L = 4;
Had_pred_M1 = compress_H_to_M(H_pred, 1);
Had_pred_M4 = compress_H_to_M(H_pred, 4);
Had_pred_M10 = compress_H_to_M(H_pred, 10);
HF_pred=compress_H_to_M(H_predf,10);


W_final_pred10 = build_precoder1(Had_pred_M10, usd, N3, L, 10);
W_final_pred4  = build_precoder1(Had_pred_M4,  usd, N3, L, 4);
W_final_pred1  = build_precoder1(Had_pred_M1,  usd, N3, L, 1);
W_final_ideal = build_precoder(Had_ideal(:,:,:,1), usd, N3, L, 0.95);
W_final_outdated =build_precoder(Had_outdated, usd, N3, L,0.95);

W_final_freq=build_precoder1(HF_pred,usd,N3,L,10);
for i = 1:length(report_slots_all)
    k = report_slots_all(i);

    % Recompute per-slot precoder
    Hk_up2date = Had_up2date(:,:,:,i);  % Use Had at slot i
    Hk_ideal   = Had_ideal(:,:,:,i);

    W_final_up2date(:,:,i) = build_precoder(Hk_up2date, usd, N3, L, 0.95);
    W_final_ideal_slotwise(:,:,i) = build_precoder(Hk_ideal, usd, N3, L, 0.95);  % For MCS matched per slot
end



delay_power = zeros(1, N3);
for l = 1:N3
    delay_power(l) = norm(squeeze(Had(:,l,:,:)), 'fro')^2;
end



%% Inputs
Nt = size(H_pred_time, 1);
Nsc = size(H_pred_time, 2);  % Num subbands
Nr = size(H_pred_time, 3);
K = size(H_pred_time, 4);    % Num time slots
numStreams = 1;

%% SNR settings
snr_db = 5:2.5:30;
snr_linear = 10.^(snr_db/10);

%% --- Compute average signal power (for normalization) ---


avg_power = mean(vecnorm(reshape(H_ideal_time, [], size(H_ideal_time,4)), 2, 1).^2);

%% --- Compute Spectral Efficiency ---
H_actual = zeros(Nt, N3, Nr, length(report_slots_all));

for i = 1:length(report_slots_all)
    k = report_slots_all(i);
    for r = 1:Nr
        H_actual(:,:,r,i) = apply_usd(Had(:,:,r,k), usd, fd);  % This ensures real-time ground truth
    end
end

eff_ideal=calcSE(snr_db,W_final_ideal,H_ideal_time,snr_linear,avg_power);
eff_pred10=calcSE(snr_db,W_final_pred10,H_pred_time,snr_linear,avg_power);
eff_pred4=calcSE(snr_db,W_final_pred4,H_pred_time,snr_linear,avg_power);
eff_pred1=calcSE(snr_db,W_final_pred1,H_pred_time,snr_linear,avg_power);
eff_freq=calcSE(snr_db,W_final_freq,H_predf,snr_linear,avg_power);
 eff_outdated = calcSE_fixed_precoder(snr_db, W_final_outdated, H_actual, snr_linear,avg_power);
eff_up2date_variable = calcSE_variable_precoder(snr_db, W_final_up2date, H_up2date, snr_linear, avg_power);
eff_ideal_variable   = calcSE_variable_precoder(snr_db, W_final_ideal_slotwise,   H_ideal_time, snr_linear, avg_power);





%% --- Plot ---
figure; hold on; grid on;

plot(snr_db, eff_ideal,             'b--s',  'LineWidth', 2, 'DisplayName', 'Ideal CSI (1st slot MCS)');
plot(snr_db, eff_ideal_variable,    'b-o',   'LineWidth', 2, 'DisplayName', 'Ideal CSI (per-slot MCS)');

plot(snr_db, eff_pred10,            'r-d',   'LineWidth', 2, 'DisplayName', 'Predicted CSI (M=10)');
plot(snr_db, eff_pred4,             'm-x',   'LineWidth', 2, 'DisplayName', 'Predicted CSI (M=4)');
plot(snr_db, eff_pred1,             'c-^',   'LineWidth', 2, 'DisplayName', 'Predicted CSI (M=1)');

plot(snr_db, eff_freq,              'k--o',  'LineWidth', 2, 'DisplayName', 'Frequency-domain AR (M=10)');

plot(snr_db, eff_up2date_variable,  'g-.p',  'LineWidth', 2, 'DisplayName', 'Up-to-date CSI');

plot(snr_db, eff_outdated,          'k:*',   'LineWidth', 2, 'DisplayName', 'Outdated CSI');

xlabel('SNR [dB]');
ylabel('Spectral Efficiency [bpcu]');
title('Spectral Efficiency vs SNR (eType-II Codebook)');
legend('Location', 'SouthEast');

