%% =========================================================
%  CSI Sample Validation Script — 500 Hz Doppler
%  Validates channel estimates BEFORE neural network training
%  =========================================================
clc; clear; close all;

%% ---- System Parameters (must match your training config) ----
txAntennaSize  = [2 2 2 1 1];   % rows, cols, pol, panels
rxAntennaSize  = [2 1 1 1 1];
rmsDelaySpread = 300e-9;         % s
maxDoppler     = 500;            % Hz  <-- your setting
nSizeGrid      = 52;             % RBs
subcarrierSpacing = 15;          % kHz
numSamplesToValidate = 50;       % number of slots to inspect

%% ---- Carrier & Channel Setup ----
carrier = nrCarrierConfig;
carrier.NSizeGrid        = nSizeGrid;
carrier.SubcarrierSpacing = subcarrierSpacing;

waveInfo        = nrOFDMInfo(carrier);
samplesPerSlot  = sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

channel = nrCDLChannel;
channel.DelayProfile        = 'CDL-C';
channel.DelaySpread         = rmsDelaySpread;
channel.MaximumDopplerShift = maxDoppler;
channel.RandomStream        = 'Global stream';
channel.TransmitAntennaArray.Size = txAntennaSize;
channel.ReceiveAntennaArray.Size  = rxAntennaSize;
channel.ChannelFiltering    = false;
channel.NumTimeSamples      = samplesPerSlot;
channel.SampleRate          = waveInfo.SampleRate;

nTx = prod(txAntennaSize);
nRx = prod(rxAntennaSize);
nSub = carrier.NSizeGrid * 12;
nSym = carrier.SymbolsPerSlot;

fprintf('=================================================\n');
fprintf('  CSI Sample Validation — 500 Hz Doppler\n');
fprintf('=================================================\n');
fprintf('Subcarriers : %d\n', nSub);
fprintf('Symbols/slot: %d\n', nSym);
fprintf('Tx antennas : %d\n', nTx);
fprintf('Rx antennas : %d\n', nRx);
fprintf('Doppler     : %d Hz\n', maxDoppler);
fprintf('Delay spread: %.0f ns\n', rmsDelaySpread*1e9);
fprintf('-------------------------------------------------\n\n');

%% =========================================================
%  CHECK 1 — Dimensions & NaN / Inf
%% =========================================================
fprintf('[CHECK 1] Dimensions, NaN, Inf ...\n');

[pathGains, sampleTimes] = channel();
pathFilters = getPathFilters(channel);
offset      = nrPerfectTimingEstimate(pathGains, pathFilters);
Hest        = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, ...
                  offset, sampleTimes);

[r, s, rx, tx] = size(Hest);
fprintf('  Hest size : [%d %d %d %d]\n', r, s, rx, tx);

expectedSize = [nSub, nSym, nRx, nTx];
if isequal([r s rx tx], expectedSize)
    fprintf('  [PASS] Dimensions correct\n');
else
    fprintf('  [FAIL] Expected [%d %d %d %d]\n', expectedSize);
end

if any(isnan(Hest(:)))
    fprintf('  [FAIL] NaN values detected!\n');
else
    fprintf('  [PASS] No NaN values\n');
end

if any(isinf(Hest(:)))
    fprintf('  [FAIL] Inf values detected!\n');
else
    fprintf('  [PASS] No Inf values\n');
end
fprintf('\n');

%% =========================================================
%  CHECK 2 — Channel Magnitude Statistics
%% =========================================================
fprintf('[CHECK 2] Channel magnitude statistics ...\n');

magH     = abs(Hest(:));
meanMag  = mean(magH);
maxMag   = max(magH);
minMag   = min(magH);
stdMag   = std(magH);

fprintf('  Mean |H| : %.4f\n', meanMag);
fprintf('  Std  |H| : %.4f\n', stdMag);
fprintf('  Max  |H| : %.4f\n', maxMag);
fprintf('  Min  |H| : %.4f\n', minMag);

if meanMag > 0 && meanMag < 10 && ~isnan(meanMag)
    fprintf('  [PASS] Magnitudes in physically reasonable range\n');
else
    fprintf('  [WARN] Magnitudes may be abnormal — check channel config\n');
end
fprintf('\n');

%% =========================================================
%  CHECK 3 — Doppler Verification (time variation across symbols)
%% =========================================================
fprintf('[CHECK 3] Doppler — time variation across symbols ...\n');

% Collect time variation over multiple slots
timeVar = zeros(numSamplesToValidate, 1);
for k = 1:numSamplesToValidate
    [pg, st]  = channel();
    pf        = getPathFilters(channel);
    off       = nrPerfectTimingEstimate(pg, pf);
    He        = nrPerfectChannelEstimate(carrier, pg, pf, off, st);
    % Variance across symbols for subcarrier 1, Rx1, Tx1
    timeVar(k) = var(abs(He(1, :, 1, 1)));
end

meanTimeVar = mean(timeVar);
fprintf('  Mean symbol-to-symbol variance : %.6f\n', meanTimeVar);

% At 500 Hz Doppler the channel MUST vary — variance should be > 0
if meanTimeVar > 1e-6
    fprintf('  [PASS] Time variation detected (Doppler active)\n');
else
    fprintf('  [FAIL] No time variation — Doppler may not be active!\n');
end

% Coherence time estimate: ~0.4/f_D
cohTime_ms = (0.4 / maxDoppler) * 1e3;
slotDur_ms = 1;   % 15 kHz SCS → 1 ms slot
fprintf('  Estimated coherence time : %.2f ms\n', cohTime_ms);
fprintf('  Slot duration            : %.2f ms\n', slotDur_ms);
if cohTime_ms < slotDur_ms * 5
    fprintf('  [WARN] Coherence time is close to slot duration — \n');
    fprintf('         time-averaging assumption may weaken\n');
else
    fprintf('  [PASS] Coherence time >> slot duration\n');
end
fprintf('\n');

%% =========================================================
%  CHECK 4 — Power Delay Profile matches CDL-C
%% =========================================================
fprintf('[CHECK 4] Power Delay Profile shape ...\n');

pdp = mean(abs(fft(Hest, [], 1)).^2, [2 3 4]);
pdp_dB = 10*log10(pdp / max(pdp));

% Energy in first half of delay taps should dominate
earlyEnergy = sum(pdp(1:round(nSub/4)));
totalEnergy  = sum(pdp);
earlyRatio   = earlyEnergy / totalEnergy;

fprintf('  Energy fraction in first 25%% of delay taps: %.2f%%\n', ...
    earlyRatio * 100);

if earlyRatio > 0.5
    fprintf('  [PASS] PDP energy concentrated at early taps (CDL-C OK)\n');
else
    fprintf('  [WARN] Energy spread — check delay spread setting\n');
end
fprintf('\n');

%% =========================================================
%  CHECK 5 — Preprocessing Sanity (truncation & normalization)
%% =========================================================
fprintf('[CHECK 5] Preprocessing checks ...\n');

Tdelay         = 1 / (nSub * carrier.SubcarrierSpacing * 1e3);
rmsTauSamples  = channel.DelaySpread / Tdelay;
truncationFactor = 10;
maxDelay       = round((channel.DelaySpread / Tdelay) * truncationFactor / 2) * 2;

fprintf('  Delay sampling period  : %.2f ns\n', Tdelay*1e9);
fprintf('  RMS delay in samples   : %.2f\n', rmsTauSamples);
fprintf('  maxDelay (truncated)   : %d samples\n', maxDelay);

Hmean  = mean(Hest, 2);
Hmean  = permute(Hmean, [1 4 3 2]);
Hdft2  = fft2(Hmean(:,:,1));

midPoint  = floor(nSub / 2);
lowerEdge = midPoint - (nSub - maxDelay) / 2 + 1;
upperEdge = midPoint + (nSub - maxDelay) / 2;
Htemp     = Hdft2([1:lowerEdge-1, upperEdge+1:end], :);
Htrunc    = ifft2(Htemp);

HtruncReal        = zeros(maxDelay, nTx, 2);
HtruncReal(:,:,1) = real(Htrunc);
HtruncReal(:,:,2) = imag(Htrunc);

fprintf('  Preprocessed array size: [%d %d %d]\n', size(HtruncReal));
expectedPre = [maxDelay, nTx, 2];
if isequal(size(HtruncReal), expectedPre)
    fprintf('  [PASS] Preprocessed dimensions correct\n');
else
    fprintf('  [FAIL] Preprocessed size mismatch\n');
end

% Normalization check
meanVal   = mean(HtruncReal(:));
stdVal    = std(HtruncReal(:));
targetStd = 0.0212;
HNorm     = (HtruncReal - meanVal) / stdVal * targetStd + 0.5;

fprintf('  Pre-norm  mean: %.4f  std: %.4f\n', meanVal, stdVal);
fprintf('  Post-norm mean: %.4f  std: %.4f  (targets: 0.5, 0.0212)\n', ...
    mean(HNorm(:)), std(HNorm(:)));

normMeanOK = abs(mean(HNorm(:)) - 0.5) < 0.01;
normStdOK  = abs(std(HNorm(:))  - targetStd) < 0.005;
if normMeanOK && normStdOK
    fprintf('  [PASS] Normalization correct\n');
else
    fprintf('  [FAIL] Normalization out of tolerance\n');
end
fprintf('\n');

%% =========================================================
%  CHECK 6 — Sample Diversity (not all samples identical)
%% =========================================================
fprintf('[CHECK 6] Sample diversity across slots ...\n');

samples = zeros(maxDelay, nTx, 2, 3);
for k = 1:3
    [pg, st] = channel();
    pf       = getPathFilters(channel);
    off      = nrPerfectTimingEstimate(pg, pf);
    He       = nrPerfectChannelEstimate(carrier, pg, pf, off, st);
    Hm       = permute(mean(He,2), [1 4 3 2]);
    Hd       = fft2(Hm(:,:,1));
    Ht_      = Hd([1:lowerEdge-1, upperEdge+1:end], :);
    Hr       = ifft2(Ht_);
    samples(:,:,1,k) = real(Hr);
    samples(:,:,2,k) = imag(Hr);
end

s1 = samples(:); 
s2 = reshape(samples(:,:,:,2),[],1);
s3 = reshape(samples(:,:,:,3),[],1);
c12 = corr(s1(1:numel(s2)), s2);
c13 = corr(s1(1:numel(s3)), s3);

fprintf('  Correlation sample1-sample2: %.4f\n', c12);
fprintf('  Correlation sample1-sample3: %.4f\n', c13);

if c12 < 0.95 && c13 < 0.95
    fprintf('  [PASS] Samples are diverse (random seed working)\n');
else
    fprintf('  [FAIL] Samples too similar — check random stream setting\n');
end
fprintf('\n');

%% =========================================================
%  CHECK 7 — PCA Compressibility
%% =========================================================
fprintf('[CHECK 7] PCA compressibility (data should compress to ~64) ...\n');

nPCA = 20;
allSamples = zeros(maxDelay * nTx * 2, numSamplesToValidate);
for k = 1:numSamplesToValidate
    [pg, st] = channel();
    pf       = getPathFilters(channel);
    off      = nrPerfectTimingEstimate(pg, pf);
    He       = nrPerfectChannelEstimate(carrier, pg, pf, off, st);
    Hm       = permute(mean(He,2), [1 4 3 2]);
    Hd       = fft2(Hm(:,:,1));
    Ht_      = Hd([1:lowerEdge-1, upperEdge+1:end], :);
    Hr       = ifft2(Ht_);
    tmp      = cat(3, real(Hr), imag(Hr));
    allSamples(:,k) = tmp(:);
end

[~, ~, ~, ~, explained] = pca(allSamples');
cumVar64 = sum(explained(1:min(64, end)));
fprintf('  Variance explained by 64 components: %.2f%%\n', cumVar64);

if cumVar64 > 80
    fprintf('  [PASS] Data is compressible — suitable for autoencoder\n');
else
    fprintf('  [WARN] Low compressibility — may need more encoded dims\n');
end
fprintf('\n');

%% =========================================================
%  PLOTS
%% =========================================================
figure('Name','Validation Plots','NumberTitle','off','Position',[100 100 1400 900]);
tiledlayout(3, 3, 'TileSpacing','compact','Padding','compact');

% Plot 1: Channel magnitude vs subcarriers
nexttile
plot(abs(Hest(:, 1, 1, 1)))
xlabel('Subcarrier'); ylabel('|H|')
title('Freq Response — Sym1, Rx1, Tx1')
grid on

% Plot 2: Time variation across symbols (Doppler)
nexttile
plot(squeeze(abs(Hest(1, :, 1, 1))))
xlabel('Symbol Index'); ylabel('|H|')
title('Time Variation — Sub1, Rx1, Tx1 (expect variation @ 500Hz)')
grid on; yline(mean(abs(Hest(1,:,1,1))),'r--','Mean')

% Plot 3: PDP
nexttile
plot(pdp_dB(1:round(nSub/2)))
xlabel('Delay Tap'); ylabel('Power (dB)')
title('Power Delay Profile (CDL-C should decay)')
grid on; ylim([-60 5])

% Plot 4: 2-D DFT (delay-angle)
nexttile
imagesc(abs(fftshift(Hdft2)))
xlabel('Tx Angle'); ylabel('Delay')
title('Delay-Angle Domain (energy should be sparse)')
colorbar; axis tight

% Plot 5: Preprocessed real part
nexttile
imagesc(real(Htrunc))
xlabel('Tx Antennas'); ylabel('Subcarriers (truncated)')
title(sprintf('Preprocessed Real Part [%dx%d]', maxDelay, nTx))
colorbar; axis tight

% Plot 6: Preprocessed imaginary part
nexttile
imagesc(imag(Htrunc))
xlabel('Tx Antennas'); ylabel('Subcarriers (truncated)')
title(sprintf('Preprocessed Imag Part [%dx%d]', maxDelay, nTx))
colorbar; axis tight

% Plot 7: Histogram pre-normalization
nexttile
histogram(HtruncReal(:), 80, 'FaceColor', [0.2 0.5 0.8])
xlabel('Value'); ylabel('Count')
title('Distribution — Pre-Normalization')
grid on

% Plot 8: Histogram post-normalization
nexttile
histogram(HNorm(:), 80, 'FaceColor', [0.2 0.7 0.4])
xlabel('Value'); ylabel('Count')
title('Distribution — Post-Normalization (center ~0.5)')
grid on; xline(0,'r--'); xline(1,'r--')

% Plot 9: PCA cumulative variance
nexttile
plot(cumsum(explained(1:min(nPCA, end))), 'o-', 'LineWidth', 1.5)
xlabel('PCA Components'); ylabel('Cumulative Variance (%)')
title('PCA Compressibility')
grid on; xline(min(64,nPCA),'r--','nEncoded=64')

%% =========================================================
%  FINAL SUMMARY
%% =========================================================
fprintf('=================================================\n');
fprintf('  VALIDATION COMPLETE\n');
fprintf('  Review [PASS]/[FAIL]/[WARN] above\n');
fprintf('  Check figures for visual confirmation\n');
fprintf('=================================================\n');