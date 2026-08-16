
function H_tf = apply_usd(Had_pred, usd, fd)
    % Converts from angle-delay to time-frequency
    H_tf = usd' * Had_pred * fd';
end
