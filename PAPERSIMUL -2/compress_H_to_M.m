function H_mod = compress_H_to_M(H, M)
    % Retains only top-M delay taps in H based on power
    
    H_mod = zeros(size(H));
    [Na, Nf, Nr, K] = size(H);  % Na = num antennas, Nf = delay taps

    % Step 1: Compute power at each delay tap
    delay_power = zeros(1, Nf);
    for l = 1:Nf
        slice = squeeze(H(:, l, :, :));  % [Na × Nr × K]
        delay_power(l) = norm(slice(:))^2;
    end

    % Step 2: Find top-M delay indices
    [~, sorted_indices] = sort(delay_power, 'descend');
    best_indices = sorted_indices(1:M);

    % Step 3: Retain only top-M delay taps
    H_mod(:, best_indices, :, :) = H(:, best_indices, :, :);
end
