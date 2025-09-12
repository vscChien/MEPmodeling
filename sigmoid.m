function y=sigmoid(x,x0,r,a)
    y=a./(1+exp(r.*(x0-x)));
end