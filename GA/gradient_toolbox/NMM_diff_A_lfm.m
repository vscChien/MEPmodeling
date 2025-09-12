function j = NMM_diff_A_lfm(para,houtput,myfunc,y_goal)

% accuracy = 1e-3;   %<-------------- hier kann mann accuracy veranderen
h = 1e-6;  


% temp = para+h;   %trick
% h = temp-para;

para1 = para+h;



%  f: 1.partialle Ableitung, Vektor T*n x p, p=Anzahl der Parameter,
%  T=Anzahl der sample, n=Anzahl der sensor, f=[df/dp1, df/dp2,...,df/dpn]
%
%                       f(x0+h)-f(x0)
%     diff(f(x0) =      --------------
%                             h
%

para_save = para;
f = zeros(length(houtput),length(para));

for i = 1:length(para)
    para_1 = para_save;
    para_1(i) = para1(i);
    
    houtput_new = myfunc(para_1,y_goal);
    f(:,i)= (houtput_new - houtput)/h;
end

j = f; %TxP

j(isnan(j))=0;
j(isinf(j))=0;

end











