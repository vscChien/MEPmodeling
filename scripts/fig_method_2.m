parentDir = fileparts(pwd); 
addpath(parentDir);

%---------fig_method_2.png----------

% Define the sigmoid function
sigmoid = @(x, x0, r, a) a ./ (1 + exp(r * (x0 - x)));
% Define x values and scale
x = 3 / 80 * (-3:3:30) + 1; % AMT average
x = x * 0.75; % AMT to RMT
z = linspace(0.55, 1.7, 1000); % Continuous range for fitting
% Data
I1 = [0, 2.609, 2.958, 10.57, 16.79, 18.82, 30.33, 30.97, 37.46, 33.35, 36.22, 36.85];
I2 = [0, 0, 0, 10.06, 9.113, 11.17, 14.89, 18.86, 22.56, 21.34, 25.60, 29.58];
I3 = [0, 0, 0, 0, 7.659, 10.90, 16.10, 15.69, 16.69, 13.47, 16.71, 21.34];
I4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11.44, 8.991];
D  = [0, 0, 0, 0, 0, 0, 0, 0, 6.389, 9.444, 10.00, 14.17];
data = {D, I1, I2, I3, I4};

colors = [64,121,184;... % blue
          238,125,0;... % orange
          79,160,14;... % green
          196,31,25;... % red
          144,103,193]/255; % purple
labels = {'D', 'I1', 'I2', 'I3', 'I4'};
figure;
tiledlayout(3,2,'TileSpacing','tight','Padding','compact');
width=10;
height=8;
set(gcf,'units','centimeters','position',[2 2 width height])
nexttile(1,[3 1])
hold on;
h=[];
for i = 1:length(data)
   y = data{i};
  
   % Scatter plot of the data
   h(i)=scatter(x, y,20,colors(i,:),'filled');
  
   % Initial parameter guesses and bounds
   p0 = [1.5, 10, 20];
   lb = [1, 0, 0];
   ub = [2.5, 500, 40];
   % Perform curve fitting
   sigmoid_fit = @(p, x) sigmoid(x, p(1), p(2), p(3));
   opts = optimset('Display', 'off');
   [p, ~] = lsqcurvefit(sigmoid_fit, p0, x, y, lb, ub, opts);
   disp(['Fitted parameters for ', labels{i}, ': ', mat2str(p)]);
   % Plot fitted curve
   plot(z, sigmoid(z, p(1), p(2), p(3)), 'Color', colors(i,:), 'LineWidth', 1.5);
end
ylim([-1 40])
legend(h,labels,'location','northwest');
xlabel('TMS intensity (RMT)');
ylabel('Amplitude (\muV)');

%========================================
dt = 0.01; % ms
t = 0:dt:(20-dt);
TMS=[0.8,1,1.5];%0.8:0.2:1.6; % in %RMT
meancurve2=[];
for i=1:length(TMS)
   meancurve2=[meancurve2,gen_DIwave(t,TMS(i))']; %1.5 intensity 
end
times=t;
%---------EP------------
d=0.1;
[EP,t]=generate_EP(d,0,1); % get EP
dt=t(2)-t(1);
EP=-EP;
%--------DIwave-------------
times2=times(1):dt:times(end); %ms
meancurve3=interp1(times,meancurve2,times2); %interp1(times,meancurve,times2);
tidx=find(times2>0);% & times2<15);
times2=times2(tidx);
padding=1000;
meancurve3=meancurve3(tidx,:);
meancurve3=[ones(padding,1)*meancurve3(1,:);meancurve3];
times3=[nan(1,padding),times2];
%--------decov(DIwave,EP)-----------
% c=conv(meancurve3,EP,'full');
% noiselevel=0.1;
% c=c+rand(size(c))*noiselevel;
lambda = 100;
rate=deconvreg(meancurve3,EP,lambda); % deconv(c,EP);
%c=conv(rate,EP,'same');%'full');
%-----------------------------------
xlimits=[0 15];
nexttile(2);
plot(times, meancurve2/max(meancurve2(:)),'linewidth',1.5);
xlim(xlimits);box off;
ylabel('Potential'); set(gca,'ytick',[0 1]);ylim([-0.1 1]);
title('DI-waves');
legend(num2str(TMS'))
nexttile(4);
plot(t-t(1), EP,'k','linewidth',1);
xlim(xlimits);box off;
ylabel('Potential'); set(gca,'ytick',[0 1]);ylim([-0.5 1]);
title(sprintf('EP (d/c=%g)',d));
nexttile(6);
plot(times3,rate/max(rate(:)),'linewidth',1.5);
xlim(xlimits);box off;
xlabel('Time (msec)');
ylabel('Firing rate');set(gca,'ytick',[0 1]);ylim([-0.1 1]);
title('Deconvoled DI-waves');

saveas(gcf,['figures' filesep 'fig_method_2.png'])
