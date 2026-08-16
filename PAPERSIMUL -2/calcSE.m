function eff = calcSE(snr_db, W_final, H, snr_linear, avg_power)
eff = zeros(size(snr_db));
[Nr, Nsc, ~, K] = size(H);
N3 = size(W_final, 2);  % Number of streams

for i = 1:length(snr_db)
    total_rate = 0;

    for k = 1:K
        for n = 1:Nsc
            H_i = squeeze(H(:,n,:,k)).';  % [Nr × Nt]
            Heff = H_i * W_final;         % [Nr × N3]
            R = Heff * Heff';             % [Nr × Nr]
            snr_eff = snr_linear(i)/avg_power * R;
            total_rate = total_rate + real(log2(det(eye(Nr) + snr_eff)));
        end
    end

    eff(i) = total_rate / (K * Nsc);  % bits/s/Hz
end