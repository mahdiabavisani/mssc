function [Z,E] = solve_mlrr(X,lambda)
% Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
% M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,?
% Information Fusion, vol. 39, pp. 168?177, 2018.
[~,numModalities] = size(X);    %number of modalities

alp=[1,1,1,1,1];
for i=1:numModalities
    X{i} = X{i}/alp(i);
end

[Z,E] = mlrra(X,X,lambda,false);
