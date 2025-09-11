parentDir = fileparts(pwd); 
addpath(parentDir);

%---------fig_method_4.png----------
figure;
t=tiledlayout(1,3,'TileSpacing','tight','Padding','compact');
width=15;
height=5;
set(gcf,'units','centimeters','position',[2 2 width height])
%-----gamma-----
nexttile(1)
alpha  = [1,3,5]; % shape
lambda = [1,1,1]*2; % scale
h=[];
legend_entries = strings(1, length(alpha));
for i=1:length(alpha) 
  x = linspace(0, 20, 1000);
  pdf_vals = pdf('gamma', x, alpha(i), lambda(i));
  h(i)=plot(x, pdf_vals,'linewidth',1.5);hold on;box off;
  legend_entries(i) = sprintf('$\\alpha=%.1f, \\lambda=%.1f$', alpha(i), lambda(i));
  xlabel('Time (msec)','fontsize',10,'FontName', 'calibri');
  ylabel('Probability density','fontsize',10,'FontName', 'calibri');
  title('Gamma');
end
legend(h, legend_entries, 'Location', 'best','Interpreter', 'latex');
nexttile(2)
alpha  = [1,1,1,1]; % shape
lambda = [1,1,1,1]*2; % scale
shifts = [20,21,22,23]; % ms
Ns     = [100,75,50,25]; % N
for i=1:length(Ns)
   spike_times = gaminv(linspace(0, 0.99, Ns(i)), alpha(i), 1/lambda(i));
   scatter(spike_times+shifts(i),1:Ns(i),20,'.');hold on;
end
set(gca,'ytick',0:25:100);
xlabel('Time (msec)','fontsize',10,'FontName', 'calibri');
ylabel('Motor unit','fontsize',10,'FontName', 'calibri');
title('Trigger time');
nexttile(3)
alpha  = [1,1,1,1]*5; % shape
lambda = [1,1,1,1]*2; % scale
shifts = [20,21,22,23]; % ms
Ns     = [100,75,50,25]; % N
h=[];
legend_entries = strings(1, length(Ns));
for i=1:length(Ns)
   spike_times = gaminv(linspace(0, 0.99, Ns(i)), alpha(i), 1/lambda(i));
   h(i)=scatter(spike_times+shifts(i),1:Ns(i),20,'.');hold on;
   legend_entries(i) = sprintf('N=%d',Ns(i));
end
legend(h, legend_entries,'fontname','calibri','fontsize',8);
xlabel('Time (msec)','fontsize',10,'FontName', 'calibri');
set(gca,'ytick',0:25:100,'yticklabel',[]);
%ylabel('Motor unit','fontsize',10,'FontName', 'calibri');
title('Trigger time');


saveas(gcf,['figures' filesep 'fig_method_4.png'])