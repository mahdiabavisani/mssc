function [Z,E] = solve_kmlrr(X,lambda)
% Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
% M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,?
% Information Fusion, vol. 39, pp. 168?177, 2018.
[Z,E] = kmlrra(X,lambda,false);


