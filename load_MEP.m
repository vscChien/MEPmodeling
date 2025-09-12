% input
%   subj: subject index 
%   iidx: intensity index 
%
% output
%   y: target signal [n x 300] if tcrop=[20,50]ms
%   t: [1 x 300]
%   yall: [n x 300 x 15 trials]
%   mep: all MEPs [N intensities x 10000 timepoints x 15 trials]
%   intensities: all intensities [1 x N]
%   times: [1 x 10000], dt=0.1ms
%
% example
%   subj=1; load_MEP(subj);
%   subj=1; iidx=[5,6];  load_MEP(subj,iidx);
function [y,t,mep,intensities,times,yall] = load_MEP(subj,iidx,tcrop,plotOn)
    if nargin < 3
        tcrop = [20 50]; %ms  
    end
    if nargin < 4
        plotOn=1;
    end
    
    TSTIM=100; % TMS stimulus was at t=100ms
    root=fileparts(mfilename("fullpath"));
    tmp=load(fullfile(root,'data_Oxford_MEP',...
             sprintf('S%d_Magstim_data',subj),...
             sprintf('S%d.mat',subj))); 
    mep=tmp.mep;
    intensities=tmp.intensities;
    times=tmp.t-TSTIM;
    if nargin < 2
        iidx=1:length(intensities); % all intensities
    end
    mep=mep(iidx,:,:);
    intensities=intensities(iidx);

    % crop   
    tidx=find(times>=tcrop(1) & times<tcrop(2));
    t = times(tidx); 
    yall=mep(iidx,tidx,:);
    y = mean(yall,3);

    % remove baseline [-20 -10]msec % especially for subject 6
    baseline=mean(mean(mep(iidx,times>=-20 & times<-10,:),3),2);
    y=y-repmat(baseline,[1,size(y,2)]);   
    mep=mep-repmat(baseline,[1,size(mep,2),size(mep,3)]);

    if plotOn
        figure % all
        for i=1:length(intensities)
          subplot(2,5,i)
          plot(times,squeeze(mep(i,:,:)),'c');hold on;grid on;
          plot(times,mean(squeeze(mep(i,:,:)),2),'k','linewidth',1.5);
          xlim([20 50])
          ylim([-2 5])
          title(sprintf('%g%% MSO',intensities(i)));
          xlabel('msec')
        end

        figure % selected
        plot(t,y,'linewidth',1.5);grid on;
        title(sprintf('Subject %d',subj))
        xlabel('Time (msec)');ylabel('Amplitude (mV)')
  
        legend_labels = strcat(string(intensities), '%MSO');
        legend(legend_labels)
    end
end