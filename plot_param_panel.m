%function plot_result(DIwave,t,p,simMEP,ref,spike_times,NRMSD,Rm,Hm,Hrc)
function plot_param_panel(p,ref)

    spike_times=ref.sim.spike_times;
    NRMSD=ref.NRMSD;
    R=ref.model.R;
    Wexc=ref.model.Wexc;
    RWinh=ref.model.RWinh; % R*Winh
    simMEP=ref.sim.simMEP2;
    DIwave=ref.model.DIwave;
    t=ref.model.t;

    nParams=length(p);
    figure;
    if isfield(ref.model,'boundary')
      boundary=ref.model.boundary;
    end
     % boundary=[zeros(nParams,1),max(1,p(:)*1.1)] ; 


    ttext=ref.model.boundarytext;
    for i=1:nParams
       subplot(3,nParams,i);hold on
       rescale=0.9;
       if isfield(ref.model,'boundary')
          locP=rescale*(p(i)-boundary(i,1))/(boundary(i,2)-boundary(i,1));% map p into 0~1
       else
          locP=0.5*rescale;
       end
       scatter(1,locP,"filled")
       text(1.3,locP,sprintf('%.2f',p(i)))
       if isfield(ref.model,'boundary')
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
    
    
    subplot(3,4,9)
    sel=[length(ref.intensities),round(length(ref.intensities)/2),1]; % select 3 TMS intensities to show
    legend_labels = strcat(string(ref.intensities(sel)), '%');
    plot(t,DIwave(sel,:)','linewidth',1.5)
    xlabel('Time (ms)')
    title('DIwaves')
    xlim([0 20]);
    ylimit=ylim; ylim([-0.05 ylimit(2)]);
    lgd=legend(legend_labels,'location','northeast');%legend(num2str(ref.intensities(sel)'),'location','northeast')
    lgd.ItemTokenSize = [5,1];

    subplot(3,4,10)
    h1=plot(1:100,R,'k.');title('Model param.');xlabel('Motor neuron')
    hold on;plot([1 10 20 60 100],R([1 10 20 60 100]),'ko'); 
    
    if ref.model.withRC
        h2=plot(1:100,Wexc,'b','linewidth',1.5);
        h3=plot(1:100,RWinh,'m','linewidth',1.5);       
        lgd=legend([h1,h2,h3],{'R','Wexc','R*Winh'});
    else
        lgd=legend(h1,'R');
    end
    lgd.ItemTokenSize = [5,1];
    ylimit=ylim; ylim([-1 ylimit(2)]);

    subplot(3,4,11)
    plot(reshape(spike_times(:,:,sel)+ref.model.axonalDelay,100*ref.model.maxES,[]),repmat(1:100,[1,ref.model.maxES]),'.');hold on;
    ylim([-0.5 100]);xlim([15 50])
    ylabel('Motor unit');
    xlabel('Time (ms)')
    title('MU trigger time')
    legend(legend_labels,'location','northeast');%legend(num2str(ref.intensities(sel)'),'location','northeast')
    
    subplot(3,4,12);hold on;box on;
    for i=sel
       [N,edges] = histcounts(spike_times(:,:,i)+ref.model.axonalDelay,15:2:50,'Normalization','countdensity');
       edges = edges(2:end) - (edges(2)-edges(1))/2;
       plot(edges, N,'linewidth',1.5);
    end
    xlim([15 50]);
    ylimit=ylim; ylim([-1 ylimit(2)]);
    lgd=legend(legend_labels,'location','northeast');%legend(num2str(ref.intensities(sel)'),'location','northeast')
    lgd.ItemTokenSize = [5,1];
    title('Histogram');xlabel('Trigger time (ms)');ylabel('Count density')
    
    set(gcf,'position',[50 50 800 600])
    
end