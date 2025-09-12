function plot_DIwave(model,ref)

    sel=[length(ref.intensities),round(length(ref.intensities)/2),1]; % select 3 TMS intensities to show
    figure;
    subplot(511);plot(model.t,model.DIwave0(sel,:)','linewidth',1); ylabel('potential');
    title(sprintf('Subject %d\nDI-waves (potential)',ref.subj));
    legend(num2str(round(ref.intensities(sel)')))
    subplot(512);plot(model.t,model.DIwave(sel,:)','linewidth',1); title('DI-waves (firing rate)');ylabel('rate');
    subplot(513);plot(model.t,[model.AMPA',model.NMDA'],'linewidth',1); title('Synaptic kernels');legend('AMPA','NMDA');ylabel('cond.');
    subplot(514);plot(model.t,model.DIwaveConv_AMPA(sel,:)','linewidth',1);title('DI-waves (AMPA conductance)');ylabel('cond.');
    subplot(515);plot(model.t,model.DIwaveConv_NMDA(sel,:)','linewidth',1);xlabel('Time (ms)');  title('DI-waves (NMDA conductance)');ylabel('cond.');
    set(gcf,'position',[50 50 400 500])
end