function plot_summary(p,ref)

    spike_times=ref.sim.spike_times;
    R=ref.model.R;
    Wexc=ref.model.Wexc;
    RWinh=ref.model.RWinh; % R*Winh
    simMEP=ref.sim.simMEP2;


    figure;
    t=tiledlayout(1,4,'TileSpacing','tight','Padding','compact');
    width=15; 
    height=4.5;
    set(gcf,'units','centimeters','position',[2 2 width height])

    nexttile(1);
    space=(max(ref.y0(:))-min(ref.y0(:)))/2;
    h1=plot(ref.t0,ref.y0+repmat((1:size(ref.y0,2))*space,[size(ref.y0,1),1]),'k','linewidth',1.5);hold on;
    h2=plot(ref.t0,simMEP+repmat((1:size(ref.y0,2))*space,[size(ref.y0,1),1]),'r','linewidth',1);
    R2=1-sumsqr(ref.y0(:)-simMEP(:))/sumsqr(ref.y0(:)-mean(ref.y0(:)));
    title(sprintf('MEP (R^2= %.2g)',R2))
    set(gca,'YColor','none','box','off','tickDir','out','ytick',[])
    xlabel('Time (ms)','fontsize',10,'FontName', 'calibri')
    xlim([15 50]);ylim([0 space*length(ref.intensities)+max(ref.y0(:))])
    for i=1:length(ref.intensities)
        text(12,i*space,sprintf('%g%%',ref.intensities(i)),'fontsize',6,'FontName', 'calibri')
    end
    ylimit=ylim;
    peak=max(ref.y0(:));
    if peak>1
        plot([1;1]*47,[-1 0]+ylimit(2),'k','linewidth',1.5);
        text(46,-0.5+ylimit(2),'1 mV','fontsize',6,'FontName', 'calibri','horizontalalignment','right')
    else
        plot([1;1]*47,[-round(peak/2,1) 0]+ylimit(2),'k','linewidth',1.5);
        text(46,-round(peak/2,1)/2+ylimit(2),sprintf('%g mV',round(peak/2,1)),'fontsize',6,'FontName', 'calibri','horizontalalignment','right')
    end

    % IO curve
    nexttile(2);
    [IO,simIO,myfit1,myfit2]=get_iocurve(simMEP,ref); 
    x=linspace(IO(1,1),IO(end,1),100);
    plot(x,myfit1(x),'k','linewidth',1); hold on;
    x=linspace(simIO(1,1),simIO(end,1),100);
    plot(x,myfit2(x),'r','linewidth',1);
    h1=scatter(IO(:,1),IO(:,2),15,'ko');
    errorbar(IO(:,1),IO(:,2),IO(:,3),'k','LineStyle','none')
    h2=scatter(simIO(:,1),simIO(:,2),15,'ro');    
    xlim([27 58]);box off;
    xlabel('TMS intensity (%MSO)','fontsize',10,'FontName', 'calibri');
    ylabel('Amplitude (mV)','fontsize',8,'FontName', 'calibri');
    title('IO curve');
    if ismember(ref.subj,[1,2,4,7,9])
        yticks=unique([get(gca,'ytick'),0.5]);
        set(gca,'ytick',yticks)
    end
    
    nexttile(3);
    if ref.model.withRC 
        h1=scatter(1:100,Wexc,10,'b.');hold on;    
        h2=scatter(1:100,RWinh,10,'m.');
    end
    h3=scatter(1:100,R,10,'k.'); hold on;
    scatter([1 10 20 60 100],R([1 10 20 60 100]),20,'ko'); 

    title('Model param.');
    ylimit=ylim;
    text(15,ylimit(2)*0.85,['R: ',sprintf('[%g, %g]',round([R(1),R(100)],1))],...
         'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none')
    if ref.model.withRC
        text(15,ylimit(2)*0.75,['Wexc: ',sprintf('[%g, %g]',round([Wexc(1),Wexc(100)],1))],...
             'color','b','fontsize',7,'FontName', 'calibri','BackgroundColor','none')
        text(15,ylimit(2)*0.65,['Winh*R: ',sprintf('[%g, %g]',round([RWinh(1),RWinh(100)],1))],...
             'color','m','fontsize',7,'FontName', 'calibri','BackgroundColor','none')
        text(30,ylimit(2)*0.55,sprintf('RCth= %g',round(p(10),1)),...
             'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none')
    end
 
    text(30,ylimit(2)*0.45,sprintf('dAxon= %g',round(ref.model.axonalDelay,1)),...
        'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none')
    text(30,ylimit(2)*0.35,sprintf('Tmu= %g',round(ref.model.Tmu,1)),...
        'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none')
    text(15,ylimit(2)*0.95,sprintf('AMPAw= %g',round(ref.model.AMPAweight,1)),...
        'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none')

    xlabel('Motor neuron','fontsize',10,'FontName', 'calibri');
    set(gca,'xtick',[1,50,100],'xticklabel',[1,50,100]);
     

    nexttile(4);
    switch 2
        case 1
            plot(reshape(spike_times+ref.model.axonalDelay,100*ref.model.maxES,[]),repmat(1:100,[1,ref.model.maxES]),'.');hold on;
            ylim([0 100]);xlim([15 50]);
            ylabel('Motor unit');set(gca,'ytick',[1,50,100],'yticklabel',[1,50,100])
        case 2 
            tmp=permute(spike_times+ref.model.axonalDelay,[1 3 2]);
            tmp=reshape(tmp,100*length(ref.intensities),ref.model.maxES); % [100*Int x maxEs]
            scatter(tmp,1:(100*length(ref.intensities)),5,'k.');hold on;
            ylim([0 100*length(ref.intensities)]);xlim([15 50]);
            plot([17 50],[1;1]*100*(1:length(ref.intensities)-1),'k-')
            for i=1:length(ref.intensities)
                text(12,i*100-50,sprintf('%g%%',ref.intensities(i)),'fontsize',6,'FontName', 'calibri')
            end
            set(gca,'YColor','none','box','off','tickDir','out','ytick',[])
    end
    plot([1;1]*47,[1,100],'k','linewidth',1.5); 
    text(45,50,'100 MUs','fontsize',8,'FontName', 'calibri','horizontalalignment','right')    
    xlabel('Time (ms)','fontsize',10,'FontName', 'calibri')
    title('MU trigger time');
    %axis square
    if isfield(ref,'figname')
        root=fileparts(mfilename("fullpath"));
        saveas(gcf,[root,filesep,ref.figname])
    end
end

%==========================================================================
function [IO,simIO,myfit1,myfit2]=get_iocurve(simMEP,ref)

    % reload single-trial MEP for std of IO curve
    [~,~,~,~,~,y0all] = load_MEP(ref.subj,ref.intensity_idx,[20,50],0);
    %-------------------------------------------
    IO=zeros(length(ref.intensity_idx),3); % [Int,amplitude,std] 
    IOall=zeros(length(ref.intensity_idx),2,size(y0all,3)); % 15 trials
    simIO=zeros(length(ref.intensity_idx),2);
    for i=1:length(ref.intensity_idx)
        tidx1=find(ref.t0>=24 & ref.t0<=28);
        [peakv1,peaki1]=max(ref.y0(tidx1,i)); %t=23.9~27.9 ms
        tidx2=find(ref.t0>=ref.t0(tidx1(peaki1)) & ref.t0<=ref.t0(tidx1(peaki1))+6); % trough within 6 ms
        [peakv2,~]=min(ref.y0(tidx2,i)); %t=23.9~27.9 ms
        IO(i,1)=ref.intensities(i);
        IO(i,2)=peakv1-peakv2;
        tmp=zeros(1,size(y0all,3)); % 15 trials
        for j=1:size(y0all,3) % 15 trials
           peakv1=max(y0all(i,tidx1,j)); %t=23.9~27.9 ms
           peakv2=min(y0all(i,tidx2,j)); % trough within 6 ms
           tmp(j)=peakv1-peakv2;
        end
        IO(i,3)=std(tmp)/sqrt(length(tmp)); % std error of MEP amplitude
        %----
        tidx=find(ref.t0>=24 & ref.t0<=28);
        [peakv1,peaki1]=max(simMEP(tidx,i)); %t=23.9~27.9 ms
        tidx=find(ref.t0>=ref.t0(tidx(peaki1)) & ref.t0<=ref.t0(tidx(peaki1))+6); % trough within 6 ms
        [peakv2,~]=min(simMEP(tidx,i)); %t=23.9~27.9 ms
        simIO(i,1)=ref.intensities(i);
        simIO(i,2)=peakv1-peakv2;
    end
    myfittype = fittype('a/(1+exp(r*(x0-x)))',...
    'dependent',{'y'},'independent',{'x'},...
    'coefficients',{'a','r','x0'});
    switch ref.subj
     case {1,2,3,4,9,10}
         [myfit1,gof,output] = fit(IO(:,1),IO(:,2),myfittype,'StartPoint',[1.4, 10, 40]);
         [myfit2,gof,output] = fit(simIO(:,1),simIO(:,2),myfittype,'StartPoint',[1.4, 10, 40]);
     case {5,7,8}
         [myfit1,gof,output] = fit(IO(:,1),IO(:,2),myfittype,'StartPoint',[1.4, 5, 50]);
         [myfit2,gof,output] = fit(simIO(:,1),simIO(:,2),myfittype,'StartPoint',[1.4, 5, 50]);
     case 6
         [myfit1,gof,output] = fit(IO(:,1),IO(:,2),myfittype,'StartPoint',[1.4, 2, 60]);
         [myfit2,gof,output] = fit(simIO(:,1),simIO(:,2),myfittype,'StartPoint',[1.4, 2, 60]);
    end
end