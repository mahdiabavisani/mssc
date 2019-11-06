
function mssc_test(Ytot,Label)
% Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
% M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,?
% Information Fusion, vol. 39, pp. 168?177, 2018.
% mssc test:
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

addpath('utility')
% normalize the data
for i=1:length(Ytot),Ytot{i}=normc(Ytot{i});end

n = max(Label);

k=0;
parm=[2 20 200 250 510 1500 5000];
for alp=parm   % cross validation: try a few parameter choices with partial data
    k=k+1;
    % Params to set:
    params.robust=true;
    params.maxIter=450;
    params.alpha_C=200;%20;
    params.alpha_E=0.0584*alp;%20;
    params.rho=1;
    params.thr=4*10^-4;
    tau     = 1;
    
    % Compute oter params
    mu_e=0;
    for i=1:length(Ytot),mu_e=mu_e+norm(Ytot{i});end
    mu_e=mu_e/length(Ytot);
    params.lambda_e=params.alpha_E/mu_e;
    
    lambda_c=0;
    for i=1:length(Ytot),lambda_c=lambda_c+computeLambda_mat(Ytot{i});end
    lambda_c=lambda_c/length(Ytot);
    params.lambda_c=lambda_c;
    
    params.tau=1;%1e-4;%2*6.03e-5;
    
    % Do the MSSC:
    [C,E,iter] =  MSSC(Ytot,params);
    
    % Evaluation:
    CKSym = BuildAdjacency(thrC(C,1));
    grps = SpectralClustering(CKSym,n);
    grps = bestMap(Label,grps);
    missrateTot(k) = sum(Label(:) ~= grps(:)) / length(Label);
    nmi_(k) = nmi(Label,grps);
    
    
end
[~,ind]= min(missrateTot);
disp(['MSSC error rate is: ',num2str(missrateTot(ind)),', nmi: ',num2str(nmi_(ind))])
disp(['alp =', num2str(parm(ind))])

end
