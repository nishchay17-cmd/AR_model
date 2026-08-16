clc
clearvars
close all

num_MC = 1;
D_max = 10;
velocities = [30, 60, 90, 120];
sgcs_all_mc = zeros(length(velocities), D_max, num_MC);

for mc = 1:num_MC
    
    % 🔁 Redefine `s` inside MC loop to re-initialize QuaDRiGa
    s = qd_simulation_parameters;
    s.center_frequency = 28e9;
    s.sample_density = 200;
    s.use_3GPP_baseline = 1;
    
    for v_idx = 1:length(velocities)
% 2. Define UE motion and slot timing
    rng(mc);  % Ensure new random seed

UE_speed_kmph =velocities(v_idx) ;
UE_speed_mps = UE_speed_kmph / 3.6;
num_slots = 200;                    % Number of slots (samples)
slot_interval = 0.1e-3;              % 1 ms per slot
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
BSAntArray = qd_arrayant.generate('3gpp-mmw', vbs, hbs, s.center_frequency, 2, ElcTltAgl_BS, 1, 1, ...
                                  Vspc_Tx_BS / s.wavelength * vbs, Hspc_Tx_BS / s.wavelength * hbs);
ueantarray = qd_arrayant.generate('3gpp-mmw', 1, 1, s.center_frequency, 1, 0, 1, 1, ...
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
slot_indices = 1:1:UEtrack.no_snapshots;     % 1 ms intervals (every 10th sample)
numSnapshots = length(slot_indices);    
 


H = zeros(Nr, Nt, N3, numSnapshots);
for j= 1:numSnapshots
    k = slot_indices(j);
   H_delay = channels(1).coeff(:,:,:,k);  % Clean and correct  % Delay-domain matrix
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



K = 50;
m = 1;
N4 = 4;
delta1 = 11;
delta2 = 2;
delta = delta1 + delta2;
N4_ext = delta + delta2 + D_max;   % = 25

T_total = size(Had, 4);
first_predicted_slot = K * m + delta + 1;
last_predicted_start = T_total - (N4 - 1);
report_slots = first_predicted_slot : last_predicted_start;

% ==========================================
% NEW UNIFIED PREDICTION AND EVALUATION LOOP
% ==========================================
L = 4; % Number of spatial beams to select
N4_ext = delta + delta2 + D_max;   

% Preallocate for this specific velocity
sgcs_list = NaN(length(report_slots), D_max);
k_idx = 1;

for k = report_slots
    % 1. Define training window up to the processing delay
    train_range = (k - delta - (K-1)) : 1 : (k - delta);  

    % 2. Train AR model. Order p=3 captures fading oscillations
    model = trainAR(Had(:,:,:, train_range), 5, 1e-5);
    
    % 3. Predict into the future
    H_pred_t = predictAR(model, N4_ext);

    % 4. Evaluate SGCS immediately for all horizons d
    for d = 1 : D_max
        k_pred = k + delta2 + d;
        if k_pred > T_total, continue; end

        % The horizon relative to the end of the training window
        n_step = delta + delta2 + d;

        % Extract Angle-Delay domain channels for this specific slot
        h_pred_ad  = H_pred_t(:,:,1,n_step); % [size(usd,2) x N3]
        h_ideal_ad = Had(:,:,1,k_pred);      % [size(usd,2) x N3]

        % Transform back to Spatial-Frequency domain inline
        h_pred_sf  = usd * h_pred_ad * fd';
        h_ideal_sf = usd * h_ideal_ad * fd';

        % Dynamic Beam Selection: Find best antennas for the ACTUAL channel right now
        pow_ref = sum(abs(h_ideal_sf).^2, 2);
        [~, idx_sorted] = sort(pow_ref, 'descend');
        best_cols = idx_sorted(1 : 2*L);

        % Compute SGCS across subcarriers
        sgcs_subc = zeros(1, N3);
        for t = 1:N3
            hp = h_pred_sf(best_cols, t);
            hi = h_ideal_sf(best_cols, t);

            % Normalize and compute trace inner product
            hp_norm = hp / norm(hp, 'fro');
            hi_norm = hi / norm(hi, 'fro');
            
            sgcs_subc(t) = abs(hp_norm' * hi_norm)^2;
        end
        sgcs_list(k_idx, d) = mean(sgcs_subc);
    end
    k_idx = k_idx + 1;
end

% Average over all valid evaluation slots for this velocity
sgcs_all_mc(v_idx, :, mc) = mean(sgcs_list, 1, 'omitnan');
% ==========================================
    end
end




  

sgcs_all = mean(sgcs_all_mc, 3);



prediction_time_ms = (1:D_max)*0.1;
figure; hold on;
plot(prediction_time_ms, sgcs_all(1,:), '-g*', 'DisplayName', '30 Km/h');
plot(prediction_time_ms, sgcs_all(2,:), '-c<', 'DisplayName', '60 Km/h');
plot(prediction_time_ms, sgcs_all(3,:), '-bs', 'DisplayName', '90 Km/h');
plot(prediction_time_ms, sgcs_all(4,:), '-ko', 'DisplayName', '120 Km/h');
xlabel('Prediction time [ms]');
ylabel('SGCS');
legend('show'); grid on; ylim([0.5 1]);


  
















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

