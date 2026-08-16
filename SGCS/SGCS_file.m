clc
clearvars
close all

num_MC =1;
D_max = 10;
velocities = [30,60,90,120];
sgcs_all_mc = zeros(length(velocities), D_max, num_MC);

for mc = 1:num_MC
    rng(mc);  % Ensure new random seed
    
    % 🔁 Redefine `s` inside MC loop to re-initialize QuaDRiGa
    s = qd_simulation_parameters;
    s.center_frequency = 4e9;
    s.sample_density = 20;
    s.use_3GPP_baseline = 1;
    
    for v_idx = 1:length(velocities)
% 2. Define UE motion and slot timing
UE_speed_kmph =velocities(v_idx) ;
UE_speed_mps = UE_speed_kmph / 3.6;
num_slots = 128;                    % Number of slots (samples)
slot_interval = 1e-3;              % 1 ms per slot
total_time = num_slots * slot_interval;

% 3. Create UE track with time-based interpolation (1 ms apart)
track_len = UE_speed_mps * total_time;
UEtrack = qd_track.generate('linear', track_len);
UEtrack.set_speed(UE_speed_mps);  % Set constant speed (in m/s)
UEtrack.interpolate('time',0.1e-3, [], [], 1);  % 1 ms time step
UEtrack.name = 'UE1';

disp(['Number of snapshots in UE track: ', num2str(UEtrack.no_snapshots)]);

% 4. Antenna array definitions (example: 8x2 BS, 1x1 UE)
hbs = 8; vbs = 2;
Hspc_Tx_BS = 0.5 * s.wavelength;
Vspc_Tx_BS = 0.5 * s.wavelength;
ElcTltAgl_BS = 0;
BSAntArray = qd_arrayant.generate('3gpp-3d', vbs, hbs, s.center_frequency, 2, ElcTltAgl_BS, 1, 1, ...
                                  Vspc_Tx_BS / s.wavelength * vbs, Hspc_Tx_BS / s.wavelength * hbs);
ueantarray = qd_arrayant.generate('3gpp-3d', 1, 1, s.center_frequency, 1, 0, 1, 1, ...
                                  Vspc_Tx_BS / s.wavelength, Hspc_Tx_BS / s.wavelength);
 

% 5. Set up layout
li = qd_layout(s);
li.no_tx = 1;
li.tx_array = BSAntArray;
li.tx_position = [0; 0; 25];
li.no_rx = 1;
li.rx_array = ueantarray;
li.rx_position = [165; 0; 1.5];
li.rx_track = UEtrack;
li.set_scenario('3GPP_38.901_UMa_NLOS');


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
for j= 1:numSnapshots
   
   H_delay = channels(1).coeff(:,:,:,slot_indices(j));  % Clean and correct  % Delay-domain matrix
    H_freq = fft(H_delay, N3, 3);                   % Convert to frequency domain
    H(:,:,:,j) = H_freq;
end
% Normalize H to unit average power
% Normalize H to unit average power
H = H / sqrt(mean(abs(H(:)).^2));  % Ensures fair power comparison
q1 = 1; q2 = 1;

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

for k = 1:numSnapshots
    H_tf = permute(H(:, :, :, k), [2 3 1]);   % [Nt × N3 × Nr]
    Had(:,:,1,k) = usd' * H_tf(:,:,1) * fd; 
end
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
H_pred_t  = zeros(Nt, N3, Nr, T_total);
H_pred_time  = zeros(Nt, N3, Nr, T_total);
H_ideal_time = zeros(Nt, N3, Nr, T_total);
for k = report_slots
    train_range = (k - delta - m*(K-1)) : m : (k - delta);
    model = trainAR(Had(:,:,:,train_range), 1,0.1);
    H_pred_temp = predictAR(model, N4); 
     for n = 0:N4-1
        k_pred = k + n; 
       
  H_pred_time(:,:,:,k_pred) = apply_usd(H_pred_temp(:,:,:,n+1), usd, fd);
 H_ideal_time(:,:,:,k_pred) = apply_usd(Had(:,:,:,k_pred), usd, fd);
     end
end

L=4;
k0 = report_slots(1);
ref_ideal = squeeze(H_ideal_time(:,:,:,k0));   % [Nt x Nsc x Nr]
power_vec = sum(sum(abs(ref_ideal).^2, 3), 2); % [Nt x 1], sum over sc and Nr
[~, idx] = sort(power_vec, 'descend');
best_cols = idx(1:2*L);

for d = 0 : D_max
    sgcs_list = [];

    for k = report_slots
        k_pred = k + delta2 + d;
        if k_pred > size(H_pred_time, 4), continue; end

        h_pred_full  = squeeze(H_pred_time(:,:,:,k_pred));
        h_ideal_full = squeeze(H_ideal_time(:,:,:,k_pred));
        sgcs_subc = zeros(1, size(h_pred_full, 2));

        for t = 1:size(h_pred_full, 2)
            H_pred_sc = squeeze(h_pred_full(:,t,:));
            H_ideal_sc = squeeze(h_ideal_full(:,t,:));
            h_pred  = H_pred_sc(best_cols, :).';
            h_ideal = H_ideal_sc(best_cols, :).';
            h_p = h_pred / norm(h_pred, 'fro');
            h_i = h_ideal / norm(h_ideal, 'fro');
            sgcs_subc(t) = abs(trace(h_p' * h_i))^2;
        end

        sgcs_list(end+1) = mean(sgcs_subc);
    end
    sgcs_all_mc(v_idx, d+1, mc) = mean(sgcs_list);  % d+1 for MATLAB indexing
end
    end
end
sgcs_all = mean(sgcs_all_mc, 3);
prediction_time_ms = 0:D_max;  % Start from 0

figure; hold on;
plot(prediction_time_ms, sgcs_all(1,:), '-g*', 'DisplayName', '30 Km/h');
plot(prediction_time_ms, sgcs_all(2,:), '-c<', 'DisplayName', '60 Km/h');
plot(prediction_time_ms, sgcs_all(3,:), '-bs', 'DisplayName', '90 Km/h');
plot(prediction_time_ms, sgcs_all(4,:), '-ko', 'DisplayName', '120 Km/h');
xlabel('Prediction time [ms]');
ylabel('SGCS');
legend('show'); grid on;
















function ua1 = ugen(hbs)
    for t = 0:hbs-1
        for j = 0:hbs-1
            ua1(t+1,j+1) = exp(1i*2*pi*t*j/hbs);
        end
    end
   ua1= ua1/sqrt(hbs);
end

function v = vgen(q,n)
    for t = 0:n-1
        v(t+1,1) = exp(1i*2*pi*t*q/(2*n));
    end
end

