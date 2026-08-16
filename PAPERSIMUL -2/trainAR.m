function model = trainAR(H_train, p, lambda)
    % Per-element scalar complex AR — avoids 173K-parameter underdetermination
    % H_train : [Na x Nf x 1 x K]
    % p       : AR order (use 1)
    % lambda  : L2 regularization (use 0.1)

    if nargin < 3, lambda = 0.1; end

    [Na, Nf, ~, K] = size(H_train);
    N = Na * Nf;
    Y = reshape(H_train, N, K);   % [N x K], each row = one bin's time series

    A_coeff = zeros(N, p);        % [N x p] — one scalar AR coeff per bin per lag

    for n = 1:N
        y = Y(n, :).';            % [K x 1] complex time series for this bin

        % Regression matrix: Phi(t-p, :) = [y(t-1), y(t-2), ..., y(t-p)]
        Phi = zeros(K-p, p);
        for i = 1:p
            Phi(:, i) = y(p+1-i : K-i);   % lag i in column i
        end
        target = y(p+1 : K);               % [K-p x 1]

        % Regularized normal equations (well-conditioned: p x p system)
        A_coeff(n, :) = ((Phi' * Phi + lambda * eye(p)) \ (Phi' * target)).';
    end

    model.A     = A_coeff;               % [N x p]
    model.order = p;
    model.state = Y(:, end-p+1 : end);   % [N x p], col p = most recent
    model.size  = [Na, Nf];
end