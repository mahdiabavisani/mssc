function [C,E,J] =  KMSSC(Y,params)
% Mahdi Abavisani, Rutgers University . mahdi.abavisani@rutgers.edu
% M. Abavisani and V. M. Patel, ?Multimodal sparse and low-rank subspace clustering,?
% Information Fusion, vol. 39, pp. 168?177, 2018.
close all;
% Parameters
tau     = params.tau;
alpha_C = params.alpha_C;
alpha_E = params.alpha_E;
lambda_c= params.lambda_c;
lambda_e= params.lambda_e;
maxIter = params.maxIter;
robust  = params.robust;
%thr= 27.5*10^-4;
thr= params.thr;

gamma = 1.3;%alpha_C / norm(Y{1},1);
mu1 = alpha_C/lambda_c;
%alpha_C * 1/computeLambda_mat(Y{1});
mu2 = alpha_E * 1;

% Extractin dimensions of Y
[~,numModalities] = size(Y);    %number of modalities
[~,numSamples] = size(Y{1});    % number of samples

for i=1:numModalities
    featureDim(i)=size(Y{i},1); % dimension of features in each modality
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Compute inverse of Sum (YY - alpha_C*I) for updating C. This term is
%%%% same for robust version and exact samples.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sum_YY=zeros(numSamples,numSamples);

for i = 1:numModalities
    YY=Y{i};
    Sum_YY =Sum_YY + YY;
end

inv_Sum_YY_acI=inv(mu1*Sum_YY + mu2*eye(numSamples)); % inverse of the term

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initial Values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C = zeros(numSamples,numSamples);
Z = zeros(numSamples,numSamples);
Ac= zeros(numSamples,numSamples);
Ae= cell(1,numModalities);
E = cell(1,numModalities);
U = cell(1,numModalities);

% for i = 1:numModalities
%     E{i} = zeros(featureDim(i),numSamples);
%     U{i} = zeros(featureDim(i),numSamples);
%     Ae{i}= zeros(featureDim(i),numSamples);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err1 = 10*thr;
i=1;
%while ( (err1 > thr) || ((i < maxIter) && (err1 > thr2)))
%while (i < 61)
while ( (err1 > thr ) && i < maxIter )
    % Update C:
    C = updateC(Sum_YY,Ac,Z,inv_Sum_YY_acI,tau,alpha_C,numModalities,numSamples,robust,mu1,mu2);
    
    %Update U & E & Ae:
    %[E,U,Ae] = updateE(Y,C,E,U,Ae,alpha_E,lambda_e,numModalities,mu1,mu2,gamma);
    %imagesc((reshape(Y{1}(:,22)-E{1}(:,22),48,42)));
    %pause(.1);
    
    %Update Z:
    Z = updateZ(C,Ac,alpha_C,lambda_c,mu1,mu2);
    
    %Update Ac:
    Ac = Ac + mu2*(C-Z);
    
    % error
    err1= errorCoef(Z,C);
    
    %     tmp=Y{1}*C;
    %     subplot(1,2,1),imagesc(reshape(E{1}(:,10),192,168)),pause(.01)
    %     subplot(1,2,2),imagesc(reshape(tmp(:,10),192,168)),pause(.01)
     J(i)=updateJ(Y,C,E,mu1,mu2,lambda_c,i,numModalities);
    i = i + 1;
    mu1=gamma*mu1;
    mu2=gamma*mu2;
    
end
fprintf('err1: %2.4f, iter: %3.0f \n',err1,i);

%%%%%%%%%%%%%%%%%%%%%%%%%%% end of main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% $$$ Y-E - robust
%%%% Updating C:
function C = updateC(Sum_YY,Ac,Z,inv_Sum_YY_acI,tau,alpha_C,numModalities,numSamples,robust,mu1,mu2)



C = inv_Sum_YY_acI*(mu1*Sum_YY + mu2*(Z-Ac/mu2));
C=C-diag(diag(C));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
%%%% Updating E, U and Ae:
function [E,U,Ae] = updateE(Y,C,E,U,Ae,alpha_E,lambda_E,numModalities,mu1,mu2,gamma)

for i = 1:numModalities
    
    E{i} = (Y{i} - Y{i}*C + alpha_E*(U{i} - Ae{i}/alpha_E))/( 1 + alpha_E);
    U{i} = max(0,(abs(E{i}+Ae{i}/alpha_E) - 1/alpha_E)) .* sign(E{i}+Ae{i}/alpha_E);
    Ae{i} = Ae{i} + alpha_E*(E{i}-U{i});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Updating Z:
function Z = updateZ(C,Ac,alpha_C,lambda_C,mu1,mu2)
Z = max(0,(abs(C+ Ac/mu2) - 1/mu2)) .* sign(C+ Ac/mu2);

Z=Z-diag(diag(Z));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Updating J:
function J = updateJ(Y,C,E,mu1,mu2,lambda_c,ii,numModalities)


first_term=0;
for i = 1:numModalities
    first_term_temp=norm(Y{i}-Y{i}*C);
    first_term =first_term + first_term_temp;
end

second_term=sum(sum(abs(C)));

third_term=0;
% for i = 1:numModalities
%     third_term_temp=sum(sum(abs(E{i})));
%     third_term =third_term + third_term_temp;
% end

J= mu1*first_term + second_term + third_term;
JJ= [first_term , second_term , third_term];

% hold on
% figure(1)
% subplot(1,3,1),plot(ii,JJ(1),'Marker','diamond','LineWidth',2),pause(0.01)
% xlabel('Iterations');
% ylabel('\Sigma ||Y^i-Y^iC-E^i||_{F}');
% title('\Sigma ||Y^i-Y^iC-E^i||_{F}''s value in iterations.')
%
% hold on
% subplot(1,3,2),plot(ii,JJ(2),'Marker','diamond','LineWidth',2),pause(0.01)
% xlabel('Iterations');
% ylabel('||C||_1');
% title('||C||_1''s value in iterations.')
%
% hold on
% subplot(1,3,3),plot(ii,JJ(3),'Marker','diamond','LineWidth',2),pause(0.01)
% xlabel('Iterations');
% ylabel('\Sigma ||E^i||_1');
% title('\Sigma ||E^i||_1''s value in iterations.')
% 
% hold on
% figure(2)
% plot(ii,J,'Marker','diamond','LineWidth',2),pause(0.01)
% xlabel('Iterations');
% ylabel('Objective function');

%ylabel('\tau \Sigma ||Y^i-Y^iC-E^i||_{F} + \lambda_c ||C||_1 + \lambda_e \Sigma ||E^i||_1');
%title('Objective function''s value in iterations.')



