%==========================================================================
% t: timepoints 
% intensity: RTM
function DIwave = gen_DIwave(t,intensity)
%     t=linspace(2, 14, 1000); 
%     intensity =1.5; % RTM
    t0   = 5;  % offset
    T    = 1.5; % peak interval
    width= 0.25; % wave width

    DIwave=zeros(size(t));

    % D, I1, I2, I3, I4
    x0 = [1.36192637, 1.04127548, 1.16603639, 1.03733872, 1.45405986];
    r =  [18.50774852, 9.26210842, 5.91559859, 17.7805388, 425.51252596];
    a =  [0.34532065, 1.0, 0.80577286, 0.46054753, 0.27828232];

    for i=1:5
        DIwave = DIwave+ exp(-(t-t0-(i-1)*T).^2/2/width^2)*sigmoid(intensity,x0(i),r(i),a(i));
    end

    if 0
        figure;
        plot(t,DIwave)
        xlabel('ms')
        ylabel('normalized amplitude')
        title(sprintf('DIwave (at %g RMT)',intensity))
    end
end
%==========================================================================
function y=sigmoid(x,x0,r,a)
    y=a./(1+exp(r.*(x0-x)));
end