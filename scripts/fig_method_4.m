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
shape  = [1,3,5]; % shape
rate = [1,1,1]*0.5; % rate
h=[];
legend_entries = strings(1, length(shape));
for i=1:length(shape) 
  x = linspace(0, 30, 1000);
  pdf_vals = pdf('gamma', x, shape(i), 1/rate(i));
  h(i)=plot(x, pdf_vals,'linewidth',1.5);hold on;box off;
  legend_entries(i) = sprintf('$\\alpha=%.1f, \\lambda=%.1f$', shape(i), rate(i));
  xlabel('Time (ms)','fontsize',10,'FontName', 'calibri');
  ylabel('Prob. density','fontsize',10,'FontName', 'calibri');
  title('Gamma');
end
lgd=legend(h, legend_entries, 'Location', 'best','Interpreter', 'latex');
lgd.ItemTokenSize = [5,1];

nexttile(2)
shape  = [1,1,1,1]; % shape
rate = [1,1,1,1]*0.5; % rate
shifts = [20,21,22,23]; % ms
Ns     = [100,75,50,25]; % N
legend_entries = strings(1, length(Ns));
h=[];
for i=1:length(Ns)
   spike_times = gaminv(linspace(0, 0.99, Ns(i)), shape(i), 1/rate(i));
   h(i)=scatter(spike_times+shifts(i),1:Ns(i),20,'.');hold on;
   legend_entries(i) = sprintf('N=%d',Ns(i));
end
lgd=legend(h, legend_entries,'fontname','calibri','fontsize',8);
lgd.ItemTokenSize = [5,1];
set(gca,'ytick',0:25:100);xlim([20 50])
xlabel('Time (ms)','fontsize',10,'FontName', 'calibri');
ylabel('Motor unit','fontsize',10,'FontName', 'calibri');
title('Trigger time');

nexttile(3)
shape  = [1,1,1,1]*5; % shape
rate = [1,1,1,1]*0.5; % rate
shifts = [20,21,22,23]; % ms
Ns     = [100,75,50,25]; % N

for i=1:length(Ns)
   spike_times = gaminv(linspace(0, 0.99, Ns(i)), shape(i), 1/rate(i));
   scatter(spike_times+shifts(i),1:Ns(i),20,'.');hold on;
end
xlabel('Time (ms)','fontsize',10,'FontName', 'calibri');
set(gca,'ytick',0:25:100,'yticklabel',[]);xlim([20 50])
title('Trigger time');


saveas(gcf,['figures' filesep 'fig_method_4.png'])