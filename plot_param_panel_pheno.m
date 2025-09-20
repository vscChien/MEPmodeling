function plot_param_panel_pheno(p,ref)

    simMEP=ref.sim.simMEP2;
    spike_times=ref.sim.spike_times;
    NRMSD=ref.NRMSD;

    nParams=length(p);
    p(4:end)=round(p(4:end)); % number of recruited MNs
    figure;
    if isfield(ref,'boundary')
      boundary=ref.boundary;
    end

    ttext=ref.boundarytext;
    for i=1:nParams
       subplot(3,nParams,i);hold on
       rescale=0.9;
       if isfield(ref,'boundary')
          locP=rescale*(p(i)-boundary(i,1))/(boundary(i,2)-boundary(i,1));% map p into 0~1
       else
          locP=0.5*rescale;
       end
       scatter(1,locP,"filled")
       if i<=3
           text(1.3,locP,sprintf('%.2f',p(i)));
       else
           text(1.3,locP,sprintf('%d',p(i)));
       end
       if isfield(ref,'boundary')
           scatter([1,1],[0,rescale],"k_")
           plot([1,1],[0,rescale],'k')
           text(1,0,num2str(boundary(i,1)),'HorizontalAlignment','center','VerticalAlignment','top')
           text(1,rescale,num2str(boundary(i,2)),'HorizontalAlignment','center','VerticalAlignment','bottom')     
       end
       ylim([-0.1 1.1])
       axis off
       title(ttext{min(i,length(ttext))})
    end

    subplot(312);
    plot(ref.y0(:),'k','linewidth',2);hold on;
    plot(simMEP(:),'r','linewidth',1);
    R2=1-sumsqr(ref.y0(:)-simMEP(:))/sumsqr(ref.y0(:)-mean(ref.y0(:)));
    fprintf('R^2 = %g\n',R2)
    title(sprintf('MEPs of subject %d \n(R^2 = %.2g, NRMSD = %.2g%%)',ref.subj,R2,NRMSD*100))

    plot([1;1]*(0:size(simMEP,2)-1)*size(simMEP,1),ylim,'k')
    xlim([0 length(simMEP(:))])
    xlabel(sprintf('Time (%g-%g ms)',ref.tcrop))
    set(gca,'xtick',(1:size(simMEP,2)-1)*size(simMEP,1),'xticklabel',[]);
    ylimit=ylim;
    for i=1:length(ref.intensities)
        text(length(ref.t0)*(i-0.5),ylimit(2)*0.8,sprintf('%d%%\nMSO',ref.intensities(i)),'horizontalalignment','center','fontsize',8)
    end
    lgd=legend('MEP','simMEP','location','west');
    lgd.ItemTokenSize = [5,1];
    ylabel('Amplitude (mV)');

    nItensities=length(ref.intensities);
    sel=[nItensities,round(nItensities/2),1]; % select 3 TMS intensities to show
    legend_labels = strcat(string(ref.intensities(sel)), '%');

    subplot(3,4,10)
    t = linspace(0, 30, 2000);
    shape=ref.sim.shape;
    rate=ref.sim.rate;
    tshift=ref.sim.tshift;
    delay=ref.sim.axonalDelday;
    pdf_vals = pdf('gamma', t, shape, 1/rate);
    for i=1:length(sel)
        plot(t+delay+(nItensities-sel(i))*tshift, pdf_vals,'linewidth',1.5);hold on;
    end
    title('Model param.');
    ylabel('Probability density');xlabel('Motor neuron');
    xlabel('Time (ms)');
    ylimit=ylim; ylim([-ylimit(2)/10 ylimit(2)]);xlim([15 50]);
    text(25,ylimit(2)*0.9,sprintf('shape=%g',round(shape,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    text(25,ylimit(2)*0.8,sprintf('rate=%g',round(rate,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    text(25,ylimit(2)*0.7,sprintf('dAxon=%g',round(delay,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    text(25,ylimit(2)*0.6,sprintf('tLag=%g',round(tshift,2)),'color','k','fontsize',7,'FontName', 'calibri','BackgroundColor','none');
    

    
    subplot(3,4,11)
    for i=sel %1:length(spike_times) % nIntensities
        plot(spike_times{i}+ref.t0(1),1:length(spike_times{i}),'.'); hold on;
    end
    lgd=legend(legend_labels,'location','northwest');
    lgd.ItemTokenSize = [5,1];
    ylim([-0.5 100]);xlim([15 50]);
    ylabel('Motor unit');
    xlabel('Time (ms)')
    title('MU trigger time');

    
    subplot(3,4,12);hold on;box on;
    for i=sel
       [N,edges] = histcounts(spike_times{i}+ref.t0(1),15:2:50,'Normalization','countdensity');
       edges = edges(2:end) - (edges(2)-edges(1))/2;
       plot(edges, N,'linewidth',1.5);
    end
    xlim([15 50]);
    ylimit=ylim; ylim([-1 ylimit(2)]);
    lgd=legend(legend_labels,'location','northeast');%legend(num2str(ref.intensities(sel)'),'location','northeast')
    lgd.ItemTokenSize = [5,1];
    title('Histogram');xlabel('Trigger time (ms)');ylabel('Count density')
    
    set(gcf,'position',[50 50 800 600])

    if 0
        %--------------------------------------------
        % estimated input strength
        subplot(3,2,5);
        for i=1:length(spike_times) % nIntensities
            Vth=1; I=Vth./(1-exp(-spike_times{i}));
            plot(1:length(spike_times{i}),I,'.');hold on;
        end
        grid on; ylabel('I_{in}');xlabel('Motor neuron');
        title('I = Vth/(1-ext(-t))')
        subplot(3,2,6);
        for i=1:length(spike_times) % nIntensities
            Vth=1; I=Vth./(1-exp(-spike_times{i})-spike_times{i}.*exp(-spike_times{i}));
            plot(1:length(spike_times{i}),I,'.');hold on;
        end
        grid on;
        ylabel('I_{in}');xlabel('Motor neuron');
        title('I = Vth/(1-ext(-t)-t*exp(-t))')
    end


end