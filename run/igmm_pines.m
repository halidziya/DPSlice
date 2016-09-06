load experiments/pines/Indian_pines_corrected.mat
load experiments/pines/Indian_pines_gt
X=reshape(indian_pines_corrected,145*145,200);
X=igmm_normalize(X);
Y=reshape(indian_pines_gt,145*145,1);
subplot(1,2,1)
scatter(X(:,1),X(:,2),5,Y);

D=size(X,2);
prefix = 'experiments/pines/';
%Prior configuration
Psi = eye(D)/10;
mu0 = zeros(1,D);
m   = D+2;
k0  = 1;
gamma = 1;

%File names
initiallabels = kmeans(X,20);
data=[prefix,'pines.matrix'];
meanp=[prefix,'pines_mean.matrix'];
psip=[prefix,'pines_psi.matrix'];
params=[prefix,'pines_params.matrix'];
initial=[prefix,'pines_initial.matrix'];
NITER = '500';
BURNIN = '400';
NSAMPLE = '10';

%Call
igmm_createBinaryFiles([prefix '/pines'],X,Psi,mu0,m,k0,gamma);
writeMat(initial,initiallabels,'double');
cmd = ['dpsl.exe ',data,' ',meanp,' ',psip,' ',params,' ',NITER,' ',BURNIN,' ',NSAMPLE,' ',initial];
fprintf(1,[cmd , '\n']);
tic;
system(cmd);
elapsed = toc;

%Combine multiple label samples
prediction=readMat([data '.labels']);
likelihoods=readMat([data '.likelihood']);
predlabs = align_labels(prediction');

%Plot
subplot(1,2,2)
scatter(X(:,1),X(:,2),4,predlabs);
title('Estimated');

%F1 scores
evaluationTable(Y,predlabs)
figure;
subplot(1,2,1);
imshow(reshape(Y,145,145),distinguishable_colors(length(unique(Y))))
subplot(1,2,2)
imshow(reshape(predlabs,145,145),distinguishable_colors(length(unique(predlabs))))