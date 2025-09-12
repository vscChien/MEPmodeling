
function plot_TMScondition(ref,idx)

    spike_times=ref.sim.spike_times;
    spike_times2=ref.sim.spike_times2;
    simMEP=ref.sim.simMEP2;
    DIwave=ref.model.DIwave;
    t=ref.model.t;
    gexc_all=ref.sim.gexc_all; % conductance (Cortex -> MN1)
    ginh_all=ref.sim.ginh_all; % conductance (RC -> MN1)
    Rr_all=ref.sim.mRC_all;
    Vr_all=ref.sim.vRC_all;

    %-------------------------
    figure('name','plot_result_detail');
    tiledlayout(6,1,'TileSpacing','tight','Padding','compact');
    width=10; height=15;
    set(gcf,'units','centimeters','position',[2 2 width height])

    nexttile();%subplot(511);
        plot(t,DIwave(idx,:),'k','linewidth',1);hold on;ylim([-0.1 1.1])
        ylimit=ylim;
        text(10,ylimit(2)*0.7,sprintf('%d%% MSO',ref.intensities(idx)),'fontsize',8,'FontName', 'calibri')
        title(sprintf('DI-waves (subject %d)',ref.subj)); 
        box off
        ylabel('Rate (Hz)','fontsize',10,'FontName', 'calibri');
        %yyaxis right; 
        %area(t,g_all(idx,:)','EdgeColor','none','FaceColor',[1 1 1]*0.7,'FaceAlpha',0.5);
        %ylabel('EPSP(mV)','fontsize',10,'FontName', 'calibri');box off;

    nexttile();%subplot(512);     
        area(t,gexc_all(idx,:)','EdgeColor','none',...
                                'FaceColor',[1 1 1]*0.5,'FaceAlpha',0.5); % input to MN1
        hold on;
        area(t,ginh_all(idx,:),'EdgeColor','none',...
                                'FaceColor',[109, 158, 235]/255,'FaceAlpha',0.5); % RC to MN1
        
        ylimit=ylim;
        text(1.5,ylimit(2)*0.7,sprintf('AMPA:NMDA \n= %g : %g',...
                               round(ref.model.AMPAweight,2),1-round(ref.model.AMPAweight,2)),...
                               'fontsize',8,'FontName', 'calibri')
        title('Effective conductances of MN1');
        legend('g_{exc}/g_{leak}','g_{inh}/g_{leak}')
        box off

    nexttile();%subplot(513);    
        scatter(spike_times2{idx}(:,2),spike_times2{idx}(:,1),20,'k.');
        xlim([0 50]);ylim([0 100]);set(gca,'ytick',[1,50,100]);
        title('MN spikes');
        ylabel('MN','fontsize',10,'FontName', 'calibri');   
        %yyaxis right; 
        %area(t,Vr_all(idx,:)','EdgeColor','none','FaceColor',[109, 158, 235]/255,'FaceAlpha',0.5); % a temporary fix
        %ylabel('EPSP(mV)','fontsize',10,'FontName', 'calibri');box off;

    nexttile();%subplot(514);
        h1=plot(t(1:(end-1)),Rr_all(idx,1:(end-1))','r','linewidth',1);title('RC activity');
        ylabel('Rate (Hz)','fontsize',10,'FontName', 'calibri'); ylim([-0.1 1.1])
        yyaxis right; 
        h2=area(t,Vr_all(idx,:)','EdgeColor','none','FaceColor',[234, 153, 153]/255,'FaceAlpha',0.5); 
        ylabel('EPSP(mV)','fontsize',10,'FontName', 'calibri');box off;
        legend([h2,h1],{'EPSP','rate'})

    nexttile();%subplot(515);
        scatter(spike_times(:,:,idx)+ref.model.axonalDelay,1:100,20,'k.');ylim([0 100])
        set(gca,'ytick',[1,50,100]);
        hold on;
        ylimit=ylim;
        text(1.5,ylimit(2)*0.7,sprintf('Axonal delay \n= %.2g ms',ref.model.axonalDelay),'fontsize',8,'FontName', 'calibri')
        title('MU trigger times');
        xlim([0 50]);ylim([0 100]);
        ylabel('MU','fontsize',10,'FontName', 'calibri');%ylabel('Rate (Hz)');

    nexttile();%subplot(516);
        plot(ref.t0,ref.y0(:,idx),'k','linewidth',2);hold on;
        plot(ref.t0,simMEP(:,idx),'r','linewidth',1.5);  
        legend(sprintf('%d%%MSO',ref.intensities(idx)),'simMEP','location','northwest')    
        R2=cal_R2(ref.y0(:,idx), simMEP(:,idx));
        NRMSD=cal_NRMSD(ref.y0(:,idx), simMEP(:,idx));
        title(sprintf('MEP (R^2 = %.2g, NRMSD = %.2g%%)',R2,NRMSD*100));box off;
        %hold on; plot([0;50],[0;0],'k','linewidth',1)
        xlabel('Time (ms)','fontsize',10,'FontName', 'calibri');xlim([0 50]);
        ylabel('Amplitude (mV)','fontsize',10,'FontName', 'calibri')

end