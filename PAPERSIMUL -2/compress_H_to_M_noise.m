function H_mod = compress_H_to_M_noise(H, keep_indices, SNR_dB)
% Adds small noise to selected delay taps to increase numerical difference
% H: [Na x Nf x Nr x K] predicted channel in Had domain
% keep_indices: vector of delay indices to retain (e.g., [13] or 1:M)
% SNR_dB: Signal-to-noise ratio to control how much noise is added

    H_mod = zeros(size(H));  % Initialize compressed channel
    signal_power = mean(abs(H(:)).^2);
    noise_power = signal_power / (10^(SNR_dB / 10));  % convert SNR to noise power

    noise = sqrt(noise_power/2) * (randn(size(H)) + 1j*randn(size(H)));  % complex Gaussian noise

    % Copy only selected delay taps and add small noise
    H_mod(:, keep_indices, :, :) = H(:, keep_indices, :, :) + noise(:, keep_indices, :, :);
end
