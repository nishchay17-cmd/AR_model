function eff = calcSE_variable_precoder(snr_db, W_all, H_all, snr_linear, avg_power)
eff = zeros(size(snr_db));
[Nr, Nsc, Nt, K] = size(H_all);
N3 = size(W_all, 2);  % Number of streams

for i = 1:length(snr_db)
    total_rate = 0;
    for k = 1:K
        W_k = W_all(:, :, k);  % [Nt × N3]
        for n = 1:Nsc
            H_i = squeeze(H_all(:,n,:,k)).';  % [Nr × Nt]
            Heff = H_i * W_k;                % [Nr × N3]
            R = Heff * Heff';                % [Nr × Nr]
            snr_eff = snr_linear(i)/avg_power * R;
            total_rate = total_rate + real(log2(det(eye(Nr) + snr_eff)));
        end
    end
    eff(i) = total_rate / (K * Nsc);  % Average SE
end
