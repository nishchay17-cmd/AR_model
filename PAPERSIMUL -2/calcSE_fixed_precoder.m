function eff = calcSE_fixed_precoder(snr_db, W_final, H_actual, snr_linear, avg_power)
    eff = zeros(size(snr_db));
    [Nr, Nsc, ~, K] = size(H_actual);
    N3 = size(W_final, 2);  

    for i = 1:length(snr_db)
        total_rate = 0;

        for k = 1:K
            for n = 1:N3
                H_k = squeeze(H_actual(:, n, :, k)).';   % [Nr × Nt]
                W_n = W_final(:,n,:);                 % [Nt × N3]
                Heff = H_k * W_n;                        % [Nr × N3]
                R = Heff * Heff';                        % [Nr × Nr]
                snr_eff = snr_linear(i) / avg_power * R;
                snr_eff = max(snr_eff, 1e-6);            % Floor for numerical stability
                total_rate = total_rate + real(log2(det(eye(Nr) + snr_eff)));
            end
        end

        eff(i) = total_rate / (K * Nsc);  % Average SE over all time and subbands
    end
end