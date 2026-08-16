function W_final = build_precoder(H, usd, N3, L, M_threshold)
% Builds precoding matrix W = W1 * W2 * WF from given channel H

    % --- Step 1: Beam Selection (W1)
    beam_powers = zeros(size(H,1),1);  % across beams (spatial dimension)

    for l = 1:size(H,1)
        beam_slice = H(l,:,:,:);
        beam_powers(l) = norm(beam_slice(:), 'fro')^2;
    end

    [~, sorted_indices] = sort(beam_powers, 'descend');
    best_beams = sorted_indices(1:2*L);  % 2 polarizations
    W1 = usd(:, best_beams);

    % --- Step 2: Delay Domain Power (for W2)
    delay_power = zeros(1, N3);
    for l = 1:N3
        delay_power(l) = mean(vecnorm(squeeze(H(:,l,:,:)), 2, 1).^2);  % Mean over antennas
    end

    sorted_power = sort(delay_power, 'descend');
    cum_power = cumsum(sorted_power);
    total_power = sum(sorted_power);

    % Determine M using energy threshold
    Mv = find(cum_power >= M_threshold * total_power, 1);

    % --- Step 3: Build WF (Delay-domain DFT basis)
    F = dftmtx(N3);
    WF = F(1:Mv,:) / sqrt(N3);  % size Mv x N3

    % --- Step 4: Build W2 based on H, best beams, and M
    W2 = build_W2(H, best_beams, Mv);  % Assumes you already have this function

    % --- Final Precoding Matrix
    W_final = W1 * W2 * WF;
end
