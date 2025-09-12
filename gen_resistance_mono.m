% generate membrane resistances of 100 motoneurons (R1 > R2 >...> R100)
%
% >> R = gen_resistance_mono(p(1:5));
function R = gen_resistance_mono(p)
    p = p(:);
    n = [1,10,20,60,100]; % MN[1,10,20,60,100]
    R = flipud(cumsum(p(1:5))); % R of MN[1,10,20,60,100]
    R = interp1(n,R,1:100,'pchip'); % [1 x 100]        
end