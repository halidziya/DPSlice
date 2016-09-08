load experiments/pines/Indian_pines_corrected.mat
load experiments/pines/Indian_pines_gt
X=reshape(indian_pines_corrected,145*145,200);
Y=reshape(indian_pines_gt,145*145,1);
Yorg = Y;
X=X(Y~=0,:);
Y=Y(Y~=0,:);
X=igmm_normalize(X);
Xorg = X;
X = X(:,1:10); % Number of dimensions selected

subplot(1,2,1)
scatter(X(:,1),X(:,2),5,Y);

D=size(X,2);
prefix = 'experiments/pines/';
%Prior configuration
m   = D+3;
Psi = eye(D)*m; 
mu0 = zeros(1,D);
k0  = 1 ;
gamma = 1;

%File names
data=[prefix,'pines.matrix'];
meanp=[prefix,'pines_mean.matrix'];
psip=[prefix,'pines_psi.matrix'];
params=[prefix,'pines_params.matrix'];
NITER = '4000';
BURNIN = '2000';
NSAMPLE = '10';

%Call
igmm_createBinaryFiles([prefix '/pines'],X,Psi,mu0,m,k0,gamma);
cmd = ['dpsl.exe ',data,' ',meanp,' ',psip,' ',params,' ',NITER,' ',BURNIN,' ',NSAMPLE];
fprintf(1,[cmd , '\n']);
tic;
system(cmd);
elapsed = toc;

%Combine multiple label samples
prediction=readMat([data '.labels']);
likelihoods=readMat([data '.likelihood']);
predlabs = align_labels(prediction');

subplot(1,2,2)
scatter(X(:,1),X(:,2),8,predlabs);
title('Estimated');
evaluationTable(Y,predlabs)
drawnow;


%F1 scores
evaluationTable(Y,predlabs)
showimage = 0;
if (showimage)
    figure;
    subplot(1,2,2);
    predim = Yorg;
    predim(Yorg~=0) = predlabs+1;
    imshow(reshape(predim,145,145),distinguishable_colors(length(unique(predim))))
    subplot(1,2,1);
    imshow(reshape(Yorg,145,145),distinguishable_colors(length(unique(Yorg))))
end
