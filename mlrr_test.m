
function mlrr_test(Ytot,Label)
% Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
% M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,?
% Information Fusion, vol. 39, pp. 168?177, 2018.
% mlrr test:
%   Inputs:
%       Ytot:  A cell type variable containing modalities matrices in each
%       cell.
%               Ytot{1}= Modality_1 \in R^{D1xN}
%               Ytot{2}= Modality_2 \in R^{D2xN}
%               ...
%
%       Label:  An N-sized vector containing the labels for the data points
%               (is only used for evaluation).
%
%  Output:   The function displays the clustering performances in terms of
%            error rate and NMI.
%


addpath('utility')
% normalize the data
for i=1:length(Ytot),Ytot{i}=normc(Ytot{i});end

n = max(Label);

k=0;
parm = [.2:.1:2];
for alp=parm  % cross validation: try a few parameter choices with partial data
    k=k+1;
    % Do MLRR:
    [C,~] = solve_mlrr(Ytot,alp);
    
    % Evaluation:
    CKSym = BuildAdjacency(thrC(C,1));
    grps = SpectralClustering(CKSym,n);
    grps = bestMap(Label,grps);
    missrateTot(k) = sum(Label(:) ~= grps(:)) / length(Label);
    nmi_(k) = nmi(Label,grps);
    
    
end
[~,ind]= min(missrateTot);
disp(['MLRR error rate is: ',num2str(missrateTot(ind)),', nmi: ',num2str(nmi_(ind))])
disp(['alp =', num2str(parm(ind))])
end




