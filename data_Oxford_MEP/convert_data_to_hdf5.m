%% Example script to load data
% Details for data collection:
% The TMS stimuli were applied in blocks of 15 in increasing order from low 
% to high intensities in steps of 3% of the maximum stimulator output (MSO) 
% using a Magstim 200 stimulator. Within each block, the inter-pulse intervals 
% were varyied  between 4.25 to 5.75 seconds.
% Electromyography (EMG) was recorded from the FDI of the right hand by positioning 
% disposable neonatal ECG electrodes in a belly-tendon montage, with the ground 
% electrode over the ulnar styloid process. The EMG signals were recorded using a 
% D440 Isolated Amplifier (Digitimer, Welwyn Garden City, UK), a Micro1401 
% (Cambridge Electronic Design, Cambridge, UK), a Digitimer HumBug Noise 
% Eliminator and Signal version 7.01 (Cambridge Electronic Design), with a 
% 16-bit resolution at a 10 kHz sampling rate, an amplifier gain of 1000 
% and a 10-1000 Hz filter.

clear all; clc;

% subject = "S1";
% intensities = ["29", "32", "35", "38", "41", "44", "47", "50", "53", "56"];

% subject = "S2";
% subject = "S3";
% subject = "S4";
subject = "S9";
% subject = "S10";
intensities = ["32", "35", "38", "41", "44", "47", "50"];

% subject = "S5";
% intensities = ["32", "35", "38", "44", "47", "50", "53", "56"];

% subject = "S6";
% intensities = ["41", "44", "47", "50", "53", "56"];

% subject = "S7";
% intensities = ["35", "38", "41", "44", "47", "50", "53", "56"];

% subject = "S8";
% intensities = ["29", "32", "35", "38", "41", "44", "47", "50", "53", "56"];

for i=1:length(intensities)
    files(i) = join([subject, '_Magstim_data/', subject, '_Magstim_', intensities(i)', 'percent.mat'], '');
end

fs = 10000;     % sampling rate: 10 kHz
T = 1000;       % Recording duration (msec)
t = (0:1000/fs:T-1000/fs);
mep = zeros(length(files), 10000, 15);

for i=1:length(files)
    disp(files(i))
    mep(i, :, :) = load(files(i)).Values;    
end

clear ans
clear i
intensities = str2double(intensities);
save(join([subject, '_Magstim_data/', subject, '.mat'], ''), '-v7.3');
