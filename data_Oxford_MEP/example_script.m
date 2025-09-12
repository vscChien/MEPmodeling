% Sorkhabi, M. M., Wendt, K., O'Shea, J., & Denison, T. (2022, April 9). 
% EMG data for Pulse width modulation-based TMS: Primary Motor Cortex 
% Responses compared to Conventional Monophasic Stimuli.
%
% DOI:      https://doi.org/10.1016/j.brs.2022.06.013 
% Dataset:  https://osf.io/5ry92/
%
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

MEP_data = load('S10_Magstim_data\S10_Magstim_50percent.mat');    
Fs = 10000;     % sampling rate: 10 kHz
T = 1000;       % Recording duration (msec)
t = (0:1000/Fs:T-1000/Fs);
figure;
plot(t, MEP_data.Values(:,1));
title ('First MEP for at 41% MSO')
xlabel('msec')
ylabel('MEP (mV)')