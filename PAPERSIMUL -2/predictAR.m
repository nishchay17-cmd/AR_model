function H_pred = predictAR(model, numSteps)
    p    = model.order;
    Na   = model.size(1);
    Nf   = model.size(2);
    N    = Na * Nf;

    state_buffer = model.state;   % [N x p], col p = most recent
    A_coeff      = model.A;       % [N x p]

    H_pred = zeros(Na, Nf, 1, numSteps);

    for t = 1:numSteps
        % Vectorized per-element prediction:
        % next(n) = A(n,1)*y(t-1) + A(n,2)*y(t-2) + ...
        % state_buffer col p = most recent lag 1, col 1 = oldest lag p
        next_y = sum(A_coeff .* fliplr(state_buffer), 2);  % [N x 1]

        H_pred(:,:,1,t) = reshape(next_y, Na, Nf);

        % Shift: drop oldest col, append newest
        state_buffer = [state_buffer(:, 2:end), next_y];
    end
end