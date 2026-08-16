function W2 = build_W2(Had, best_beams, Mv)


  [Nb, N3, Nr, K] = size(Had);  % assuming [Nb x N3 x Nr x K]

delay_power = zeros(Nb, N3);
for i = 1:Nb
    for j = 1:N3
        slice = squeeze(Had(i,j,:,:));  % [Nr x K]
        delay_power(i,j) = norm(slice, 'fro')^2;
    end
end

% Then keep only the best beams
delay_power_beams = delay_power(best_beams, :);  % [2L x N3]
delay_power_sum = max(delay_power_beams, [], 1);  % Use only strongest beam per delay
    % [1 x N3]

    [~, delay_idx] = sort(delay_power_sum, 'descend');
    best_delays = delay_idx(1:Mv);

    W2 = zeros(length(best_beams), Mv);

    for l = 1:length(best_beams)
        for m = 1:Mv
            beam_idx = best_beams(l);
            delay_idx = best_delays(m);
            patch = squeeze(Had(beam_idx, delay_idx, :, :));  % [Nr x K]
            
            % Use principal component energy
            [U,~,~] = svd(patch, 'econ');
            energy = norm(U(:,1)' * patch, 'fro');
            W2(l,m) = energy;
        end
    end

    W2 = W2 / norm(W2, 'fro');
end
