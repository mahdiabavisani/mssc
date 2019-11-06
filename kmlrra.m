function [Z,E] = kmlrra(XX,lambda,display)
% Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
% M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,?
% Information Fusion, vol. 39, pp. 168?177, 2018.

% Extract dimensions of Y
[~,numModalities] = size(XX);    %number of modalities
[~,n] = size(XX{1});% number of samples
%[~,m] = size(A{1});% number of samples
%m=size(A{1},2);


E=0;

tol = 1e-8;
maxIter = 550;
%[d n] = size(X);
%m = size(A,2);
rho = 1.3;
max_mu = 1e10;
mu = 1e-6;
if nargin<4
    display = true;
end
if nargin<3
    norm_x = norm(XX{end},2);
    lambda = 1/(sqrt(n)*norm_x);
end

XxX=XX{1};
for i=2:numModalities
    XxX=XxX+XX{i};
end

% as to the inv_a = inv(A'*A+eye(m));
inv_a = inv(XxX+eye(n));

%% Initializing optimization variables
% intialize
J = zeros(n,n);
Z = zeros(n,n);

%E = sparse(d,n);

%Y1 = zeros(d,n);


Y2 = zeros(n,n);
%% Start main loop
iter = 0;
if display
    disp(['initial,rank=' num2str(rank(Z))]);
end
for iter=1:maxIter
    iter = iter + 1;
    %update J
    temp = Z + Y2/mu;
 
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    
    
    U = U(:,1:svp);sigma = sigma(1:svp);
    U = U*diag(sqrt(sigma));
    U = normr(U);
    J  = (U*U').^4;
    %udpate Z
    
    
    
    Z = inv_a*(XxX+J+(Y2)/mu);

    
    leq2 = Z-J;
    stopC = max(max(max(abs(leq2))));
    if display && (iter==1 || mod(iter,50)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-4*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol
        break;
    else
        
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end