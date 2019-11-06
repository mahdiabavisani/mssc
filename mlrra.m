function [Z,JJ] = mlrra(X,A,lambda,display)
% Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
% M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,?
% Information Fusion, vol. 39, pp. 168?177, 2018.

% Extract dimensions of Y
[~,numModalities] = size(X);    %number of modalities
[~,n] = size(X{1});% number of samples
%[~,m] = size(A{1});% number of samples
m=size(A{1},2);

for i=1:numModalities
    featureDim(i)=size(X{i},1); % dimension of features in each modality
end

tol = 1e-8;
maxIter =15;% 10e5;
%[d n] = size(X);
%m = size(A,2);
rho = 1.1;
max_mu = 1e10;
mu = 1e-6;
if nargin<4
    display = true;
end
if nargin<3
    norm_x = norm(X{end},2);
    lambda = 1/(sqrt(n)*norm_x);
end





%as to the atx = A'*X;
atx=A{1}'*X{1};
for i=2:numModalities
    atx_t=A{i}'*X{i};
    atx=atx+atx_t;
end


% as to the A'*A
AA=A{1}'*A{1};
for i=2:numModalities
    AA_t=A{i}'*A{i};
    AA=AA+AA_t;
end

% as to the inv_a = inv(A'*A+eye(m));
inv_a = inv(AA+eye(m));

%% Initializing optimization variables
% intialize
J = zeros(m,n);
Z = ones(m,n);
JJ=0;


E = cell(1,numModalities);
Y1 = cell(1,numModalities);

for i = 1:numModalities
    E{i} = zeros(featureDim(i),n);
    %U{i} = zeros(featureDim(i),n);
    Y1{i}= zeros(featureDim(i),n);
end

Y2 = zeros(m,n);
%% Start main loop
iter = 0;
if display
    disp(['initial,rank=' num2str(rank(Z))]);
end



while iter<maxIter
    iter = iter + 1;
    % mu=mu/numModalities;
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
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    %    udpate Z
    mu=mu*numModalities;
    
    % as to the A'*(E)
    AE=A{1}'*(E{1});
    for i=2:numModalities
        AE_t=A{i}'*(E{i});
        AE=AE+AE_t;
    end
    
    
    % as to the A'*(Y1)
    AY1=A{1}'*(Y1{1});
    for i=2:numModalities
        AY1_t=A{i}'*(Y1{i});
        AY1=AY1+AY1_t;
    end
    
    
    
    
    Z = inv_a*(atx-AE+J+(AY1-Y2)/mu);
    %update E
    for i=1:numModalities
        xmaz{i} = X{i}-A{i}*Z;
        temp1{i} = xmaz{i}+Y1{i}/mu;
        E{i} = solve_l1l2(temp1{i},lambda/mu);
        leq1{i} = xmaz{i}-E{i};
        
    end
    
    
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1{end}))),max(max(abs(leq2))));
    if display && (iter==1 || mod(iter,20)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-4*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol
        break;
    else
        for i=1:numModalities
            
            Y1{i} = Y1{i} + mu*leq1{i};
        end
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
    
    
    %JJ(iter)=updateJ(A,X,Z,E,mu,lambda,i,numModalities);
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

function J = updateJ(A,Y,Z,E,mu,lambda_e,ii,numModalities)
Z = A{1}*Z;
for i=2:numModalities
    Z =Z+ A{i}*Z;
end
C=Z(1:1400,1:1400);

first_term=0;
for i = 1:numModalities
    first_term_temp=norm(Y{i}-Y{i}*C-E{i});
    first_term =first_term + first_term_temp;
end
[U,sigma,V] = svd(C,'econ');
second_term = sum(diag(sigma));

third_term=0;
for i = 1:numModalities
    third_term_temp=sum(sum(abs(E{i})));
    third_term =third_term + third_term_temp;
end
disp(['Obj fn=',num2str(first_term) ,',', num2str(second_term),',', num2str(lambda_e*third_term)])
J= first_term + second_term + lambda_e*third_term;



