parentDir = fileparts(pwd); 
addpath(parentDir);

%---------fig_method_3-detail.png----------
[ref.muaps,ref.tmuap] = load_muap();
figure;
tiledlayout(10,10,'TileSpacing','tight','Padding','compact');
width=12;
height=15;
set(gcf,'units','centimeters','position',[2 2 width height])
for i=1:100
   nexttile()
   plot(ref.tmuap,ref.muaps(:,i)*1000,'b','linewidth',1.5); hold on;
   plot([0;10],[0;0],'k'); scatter([0,5,10],0,8,'k|')
  
   xlim([-2 10])
   if i<=20
       ylimits=[-1 1]*max(max(abs(ref.muaps(:,1:20))))*1.1*1000;
       ylim(ylimits)
       ytick=0.03;
   elseif i<=40
       ylimits=[-1 1]*max(max(abs(ref.muaps(:,21:40))))*1.1*1000;
       ylim(ylimits)
       ytick=0.06;
   elseif i<=60
       ylimits=[-1 1]*max(max(abs(ref.muaps(:,41:60))))*1.1*1000;
       ylim(ylimits)
       ytick=0.1;
   elseif i<=80
       ylimits=[-1 1]*max(max(abs(ref.muaps(:,61:80))))*1.1*1000;
       ylim(ylimits)
       ytick=0.2;
   elseif i<=100
       ylimits=[-1 1]*max(max(abs(ref.muaps(:,81:100))))*1.1*1000;
       ylim(ylimits)
       ytick=0.5;
   end
   if ismember(i,[1,11,21,31,41,51,61,71,81,91])
      plot([0;0],[0;ytick],'k'); scatter(0,ytick,8,'k_')
      text(0,ylimits(2)/2,[num2str(ytick),' mV'],'VerticalAlignment','bottom','fontsize',6,'FontName', 'calibri');
   end
   set(gca,'ytick',[],'xtick',[]);
   title(num2str(i))
   axis off
end
saveas(gcf,['figures' filesep 'fig_method_3-detail.png'])

%---------fig_method_3.png----------
[ref.muaps,ref.tmuap] = load_muap();
ylimits=[-1 1]*max(abs(ref.muaps(:)))*1.1*1000;
figure;
t=tiledlayout(2,5,'TileSpacing','tight','Padding','compact');
width=15;
height=7;
set(gcf,'units','centimeters','position',[2 2 width height])
for i=1:10
  nexttile()
  plot(ref.tmuap,ref.muaps(:,i*10-9:i*10)*1000,'k');
  ylim(ylimits)
  hold on
  grid on
  xlim([0 10])
  title(sprintf('%d - %d',i*10-9,i*10))
  box off
end
saveas(gcf,['figures' filesep 'fig_method_3.png'])

