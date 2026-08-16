% function model = trainAR(H_train, p)
%     [Na, Nf, ~, K] = size(H_train);
%     Y = reshape(H_train, Na * Nf, K);  % Flatten each sample to a column
% 
%     % Prepare training data
%     X_input = [];
%     X_output = [];
% 
%     for t = p+1:K
%         x_stack = [];
%         for i = 1:p
%             x_stack = [x_stack; Y(:, t - i)];  % Stack p previous vectors
%         end
%         X_input = [X_input, x_stack];     % [p*(Na*Nf) x (K-p)]
%         X_output = [X_output, Y(:, t)];   % [Na*Nf x (K-p)]
%     end
% 
%     % Solve linear system: Y_t = A * [Y_{t-1}; Y_{t-2}; ...; Y_{t-p}]
%     A = X_output * pinv(X_input); % Moore–Penrose pseudoinverse
% 
%     model.A = A;
%     model.order = p;
%     model.state = Y(:, end - p + 1 : end);  % Last p states for prediction
%     model.size = [Na, Nf];
% end
function model = trainAR(H_train, p, lambda)
    [Na, Nf, ~, K] = size(H_train);
    Y = reshape(H_train, Na * Nf, K);  % Flatten to [features × time


    %--- Data preparation ---%
    X_input = zeros(p * Na * Nf, K - p);  % Preallocate for efficiency
    X_output = zeros(Na * Nf, K - p);
    
    for t = p+1:K
        stacked = reshape(Y(:, t-p : t-1), [], 1);  % Stack p lags
        X_input(:, t-p) = stacked;  
        X_output(:, t-p) = Y(:, t);
    end

    %--- Regularized solution ---%
    if lambda > 0
        % Ridge regression: L2 regularization
        I = eye(size(X_input, 1));
        A = X_output * X_input' / (X_input * X_input' + lambda * I);
    else
        % Fallback to pseudoinverse (not recommended)
        A = X_output / X_input;  % Equivalent to pinv but faster
    end


    %--- Model storage ---%
    model.A = A;
    model.order = p;
    model.state = Y(:, end-p+1:end);  % Last p states for prediction
    model.size = [Na, Nf];
    model.lambda = lambda;  % Store regularization strength
end

