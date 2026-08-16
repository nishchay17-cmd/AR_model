%% --- Channel Aging Check (Optional, for debugging only) ---
k0 = report_slots(1);
ref = squeeze(H_ideal_time(:,:,:,k0));  % [Nt x Nsc x Nr]

max_d = D_max;
aging_metric = zeros(1, max_d + 1);  % Include d = 0

for d = 0:max_d
    k_future = k0 + d;
    if k_future > size(H_ideal_time, 4)
        continue;
    end

    target = squeeze(H_ideal_time(:,:,:,k_future));  % [Nt x Nsc x Nr]
    num = 0;

    for t = 1:size(ref, 2)
        h1 = squeeze(ref(:,t,:));
        h2 = squeeze(target(:,t,:));
        h1 = h1 / norm(h1, 'fro');
        h2 = h2 / norm(h2, 'fro');
        num = num + abs(trace(h1' * h2))^2;
    end

    aging_metric(d + 1) = num / size(ref, 2);  % Index offset for MATLAB
end

% Plot
plot(0:max_d, aging_metric, '-o', 'LineWidth', 1.5);
xlabel('Time Gap d');
ylabel('Channel Similarity');
title('Channel Aging: Similarity between h(k₀) and h(k₀ + d)');
grid on;


% Initialize NMSE storage
nmse = nan(1, T_total);

for k = report_slots
    Hp = H_pred_time(:,:,:,k);
    Hi = H_ideal_time(:,:,:,k);

    % Skip if either is zero
    if all(Hp(:) == 0) || all(Hi(:) == 0)
        continue;
    end

    % Compute NMSE (not in dB yet)
    error_power = norm(Hp(:) - Hi(:))^2;
    true_power = norm(Hi(:))^2;

    nmse(k) = error_power / true_power;
end

% Convert to dB
nmse_db = 10 * log10(nmse);

% Plot
figure;
plot(nmse_db, 'LineWidth', 1.5);
xlabel('Time Slot (k)');
ylabel('NMSE (dB)');
title('Normalized Mean Squared Error between H_{pred} and H_{ideal}');
grid on;
L = 4;
k0 = report_slots(1);
ref_ideal = squeeze(H_ideal_time(:,:,:,k0));   % [Nt x Nsc x Nr]
power_vec = sum(sum(abs(ref_ideal).^2, 3), 2); % [Nt x 1], sum over sc and Nr
[~, idx] = sort(power_vec, 'descend');
best_cols = idx(1:2*L);