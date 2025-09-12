function Y=crossover(X,n)
% choose n pairs from X, to get 2*n new crossover
% X: population
% n: no of pairs of chromosomes crossover
[x,y] = size(X); % x: population size, y: parameter size
E = zeros(2*n,y);
for i = 1:n  % crossover repeat n times
    % r: select two chromosoms for crossover
    r = randi(x,1,2); % return {1x2}, imax = x = population size 
    while r(1)==r(2)  % check make sure, the crossover should not be the same
        r=randi(x,1,2);
    end
    
    A = X(r(1),:); % chromosome 1 for crossover
    B = X(r(2),:); % chromosome 2 for crossover
    c = 1+randi(y-1); % select cut point
    D = A(1,c:y); % reserve
    A(1,c:y) = B(1,c:y);
    B(1,c:y) = D;
    % Now A and B are chromosomes after cross over
    
    E(2*i-1,:) = A;
    E(2*i,:) = B;
end
Y=E;