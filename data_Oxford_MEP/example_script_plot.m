% visualization
% based on example_script.m
clear all
close all
subj=10;
load(fullfile(sprintf('S%d_Magstim_data',subj),...
              sprintf('S%d.mat',subj))) 
%-------------------
figure
for i=1:length(intensities)
  subplot(2,5,i)
  plot(t,squeeze(mep(i,:,:)),'k');hold on;
  plot(t,mean(squeeze(mep(i,:,:)),2),'r','linewidth',2);
  xlim([110 160]);
  ylim([-2 5]);
  grid on;
  title(sprintf('%g%% MSO',intensities(i)));
  xlabel('msec')
end
%-------------------
tmp=[];
for i=1:length(intensities)
  tmp=[tmp, squeeze(mep(i,:,:))];
end
[v,idx]=sort(max(tmp),'descend');
tmp=tmp(:,idx);
%-------------------
figure;
subplot(121)
plot(t,tmp+repmat((1:size(tmp,2))*0.5,10000,1));
xlim([110 160]);
grid on;
xlabel('msec')

subplot(122)
imagesc(t,1:length(idx),tmp');
set(gca,'ydir','normal');
xlim([110 160]);
clim([-1 1]*max(tmp(:))*0.5);colorbar;
grid on;
xlabel('msec')
