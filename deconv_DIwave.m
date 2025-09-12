% input 
%    DIwave: [3 x 5000] 
%    times: [1 x 5000] 0~50 ms
%
% output
%    DIwave_deconv: [3 x 5000] 
function DIwave_deconv=deconv_DIwave(times,DIwave,ref)

    DIwave_deconv=zeros(size(DIwave));
    %---------EP------------
    if ~isfield(ref,'EP')
        d=0.1;
        [EP,t]=generate_EP(d); % get EP
        dt=t(2)-t(1);
        EP=-EP;
        %EP=EP-min(EP);
    else
        EP=ref.EP;
        dt=ref.dt;
    end
    %--------DIwave-------------  
    for i=1:size(DIwave,1) % number of intensities
        times2=times(1):dt:times(end); %ms
        DIwave2=interp1(times,DIwave(i,:),times2); %interp1(times,meancurve,times2);
        %--------decov(DIwave,EP)-----------
        lambda = 100;
        rate=deconvreg(DIwave2',EP,lambda); % deconv(c,EP);
        DIwave_deconv(i,:)=interp1(times2,rate,times);
        %c=conv(rate,EP,'same');%'full');           
    end
    DIwave_deconv=DIwave_deconv/max(DIwave_deconv(:));
%     figure;
%     subplot(211);plot(times,DIwave')
%     subplot(212);plot(times,DIwave_deconv')
end

