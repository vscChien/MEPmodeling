function plot_summary_pheno(p,ref)

    simMEP=ref.sim.simMEP2;
    spike_times=ref.sim.spike_times;

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
    x = linspace(0, 30, 2000);
    shape=ref.sim.shape;
    rate=ref.sim.rate;
    tshift=ref.sim.tshift;
    delay=ref.sim.axonalDelday;
    pdf_vals = pdf('gamma', x, shape, 1/rate);
    plot(x+delay, pdf_vals,'k','linewidth',1);
    title('Model param.');
    ylabel('Prob. density');xlabel('Motor neuron');
    xlabel('Time (ms)');
    ylimit=ylim; ylim([-ylimit(2)/10 ylimit(2)]);xlim([15 50]);
    text(25,ylimit(2)*0.9,sprintf('shape=%g',round(shape,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    text(25,ylimit(2)*0.8,sprintf('rate=%g',round(rate,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    text(25,ylimit(2)*0.7,sprintf('dAxon=%g',round(delay,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    text(25,ylimit(2)*0.6,sprintf('tLag=%g',round(tshift,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    box off



    nexttile(4);
    switch 2
        case 1
            for i=1:length(spike_times)
                plot(spike_times{i}+ref.t0(1),1:length(spike_times{i}),'.');hold on;
            end
            ylim([0 100]);xlim([15 50]);
            ylabel('Motor unit');set(gca,'ytick',[1,50,100],'yticklabel',[1,50,100])
        case 2 
            for i=1:length(spike_times)
                scatter(spike_times{i}+ref.t0(1),(1:length(spike_times{i}))+(i-1)*100,5,'k.');hold on;
            end
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