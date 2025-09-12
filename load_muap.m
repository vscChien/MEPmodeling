% muaps: [200 x 100] on 100 MUAPs in uV
% tmuap: [1 x 200], 0~20 ms, dt = 0.1 msec
function [muaps, t] = load_muap(plotOn) 
    if nargin<1
        plotOn=0;
    end
    root=fileparts(mfilename("fullpath"));

    if exist(fullfile(root,'data_MUAP','muap.mat'),'file')
        tmp=load(fullfile(root,'data_MUAP','muap.mat')); % .mat file from below .hdf5
        muaps=tmp.muaps;
        t=tmp.t;
    else
        muaps=h5read(fullfile(root,'data_MUAP','muscle_model',...
                     'Dist1_Monopolar_Rest_NormalCV_New.hdf5'),'/MUAPShapes');
        muaps=muaps';  
        tmuap = linspace(0, 20, 20001); % 0~20 ms, dt = 0.001 msec
        
        %---cut zeros (around first 2000 points)----
        idx = find(sum(abs(muaps),2)~=0,1)-1; % 2001
        muaps = - muaps(idx:end,:); % flipped
        tmuap = tmuap(idx:end);
        tmuap = tmuap - min(tmuap);
        %------------------------------------
        % downsample to dt = 0.1 msec
        dt=0.1; %ms
        t=0:dt:(20-dt);
        muaps=interp1(tmuap,muaps,t,'linear',0);
        %------------------------------------
    end
    if plotOn
        figure;
        for i=1:20
            subplot(4,5,i); 
            %plot(tmuap,1e6*muaps(:,i*5-4:i*5),'linewidth',1.5);
            plot(t,1e6*muaps(:,i*5-4:i*5),'linewidth',1.5);
            grid on;
            legend(num2str((i*5-4:i*5)'));
        end
        xlabel('ms')
        ylabel('\muV')
    end 

end
