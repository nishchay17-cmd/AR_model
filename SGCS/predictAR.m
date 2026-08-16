% function H_pred = predictAR(model, numSteps)
%     A = model.A;
%     p = model.order;
%     [Na, Nf] = deal(model.size(1), model.size(2));
% 
%     % Initialize prediction buffer
%     state_buffer = model.state;  % [Na*Nf x p]
%     H_pred = zeros(Na, Nf, 1, numSteps);
% 
%     for t = 1:numSteps
%         % Form stacked input: [y_t; y_{t-1}; ...; y_{t-p+1}]
%         stacked_input = [];
%         for j = p:-1:1
%             stacked_input = [stacked_input; state_buffer(:, j)];
%         end
% 
%         % Predict next vector
%         next_y = A * stacked_input;
% 
%         % Store reshaped output
%         H_pred(:,:,1,t) = reshape(next_y, Na, Nf);
% 
%         % Update state buffer: discard oldest, append newest
%         state_buffer = [state_buffer(:,2:end), next_y];
%     end
% end

function H_pred = predictAR(model, numSteps)
    A = model.A;
    p = model.order;
    [Na, Nf] = deal(model.size(1), model.size(2));
    D = Na * Nf;  % Flattened dimension
    
    % Preallocate state buffer (column-major)
    state_buffer = model.state;  % [D x p]
    H_pred = zeros(Na, Nf, 1, numSteps);
    
    for t = 1:numSteps
        % Vectorized stacking: reverse columns for chronological order
        stacked_input = state_buffer(:, end:-1:1); % Most recent first
        stacked_input = stacked_input(:);           % [p*D x 1]
        
        % Prediction (avoid numerical instability)
        next_y = A * stacked_input;
        
        % Handle residual amplification in continuous prediction
        if any(isnan(next_y)) || norm(next_y) > 1e6
            warning('Prediction diverging at step %d. Check model stability.', t);
            next_y = zeros(D, 1);  % Fail-safe
        end
        
        % Store and update
        H_pred(:, :, 1, t) = reshape(next_y, Na, Nf);
        state_buffer = [state_buffer(:, 2:end), next_y];  % Sliding window
    end
end
