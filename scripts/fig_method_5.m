parentDir = fileparts(pwd); 
addpath(parentDir);

%---------fig_method_5.png----------

%------ Biexponential synaptic kernels (analytical)------
labels={'In->MN(AMPA | 1ms, 5ms)',...
        'In->MN(NMDA | 3ms, 50ms)',...
        'MN->RC(AChR | 0.5ms, 3.6ms)',...
        'MN->RC(AChR | 1.8ms, 20.2ms)',...
        'RC->MN(GlyR | 1ms, 6ms)'};
tau=[1,   5;
     3,   50;
     0.5, 3.6;
     1.8, 20.2;
     1,   6];
h=[ 1.8692, 1.2731, 1.5977, 1.3908, 1.7175]; % normalizing terms


dt=0.1; %ms
tlength = 50; %ms
t = 0:dt:(tlength-dt);
kernels=zeros(5,tlength/dt);
for i=1:5
  kernels(i,:)=h(i)*(exp(-t /tau(i,2))-exp(-t/tau(i,1)));
end
figure;plot(t,kernels','linewidth',2);
ylim([0 1]);grid on;
legend(labels);
title('Biexponential synaptic kernels (analytical)')


%------ Biexponential synaptic kernels (numerical)------
kernels2=zeros(5,tlength/dt);
h2=[14.6481, 3.9770, 26.5924, 6.9913, 14.0482]; % normalizing terms
for i=1:5
  v = zeros(1,length(t));
  v2 = zeros(1,length(t));
  In = zeros(1,length(t)); In(1)=1; % impulse
  for tt=1:length(t)-1
      dv  = v2(tt);
      dv2 = h2(i)*In(tt) - (tau(i,1)+tau(i,2))*v2(tt)/tau(i,1)/tau(i,2) - v(tt)/tau(i,1)/tau(i,2); 
      v(tt+1)=v(tt)+dv*dt;
      v2(tt+1)=v2(tt)+dv2*dt;
  end
  kernels2(i,:)=v;
end
figure;plot(t,kernels2','linewidth',2);
ylim([0 1]);grid on;
legend(labels)
title('Biexponential synaptic kernels (numerical)')


saveas(gcf,['figures' filesep 'fig_method_5.png'])