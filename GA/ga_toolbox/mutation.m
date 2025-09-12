function Y = mutation(X,n)
% X: population
% n: no of pairs of chromosomes mutation
[x,y]=size(X);
E = zeros(2*n,y);
for i = 1:n
    r = randi(x,1,2);         % select two chromosmes for mutation
    while r(1)==r(2)
        r = randi(x,1,2);
    end
    
    A = X(r(1),:); % chromosome 1 for mutation
    B = X(r(2),:); % chromosome 2 for mutation
    c = randi(y);  % select cut point
    D = A(1,c); % reserve
    A(1,c) = B(1,c);  %use exchange 1 parameter for mutation
    B(1,c) = D;
    
    E(2*i-1,:) = A;
    E(2*i,:) = B;
end
Y=E;