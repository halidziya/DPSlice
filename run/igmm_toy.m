%%Exeriment Folder
experimentname = 'toy';
prefix = ['experiments/' experimentname '/'];
if ~(exist(prefix, 'dir') == 7)
    mkdir(prefix);
end

%% Generate Toy Data
NCOMP = 6;
D=2;
S=10;
NPOINTS=1000;
mus=mvnrnd(zeros(1,D),eye(D,D),NCOMP);
sigmas = zeros(D,D,NCOMP);
for i=1:NCOMP
    sigmas(:,:,i)=iwishrnd(eye(D,D)/S,D+3);
end
labels=randi(NCOMP,NPOINTS,1);
X = zeros(NPOINTS,D);
for i=1:NPOINTS
    X(i,:) = mvnrnd(mus(labels(i),:),sigmas(:,:,labels(i)));
end
subplot(1,2,1);
scatter(X(:,1),X(:,2),4,labels);
title('Original');

%% Call Executable File

%Prior configuration
Psi = eye(D)/10;
mu0 = zeros(D,1);
m   = D+3;
k0  = 1;
gamma = 1;

%File names
data=[prefix,'toy.matrix'];
meanp=[prefix,'toy_mean.matrix'];
psip=[prefix,'toy_psi.matrix'];
params=[prefix,'toy_params.matrix'];
NITER = '1000';
BURNIN = '500';
NSAMPLE = '20';

%Call
igmm_createBinaryFiles([prefix '/toy'],X,Psi,mu0,m,k0,gamma);
cmd = ['dpsl.exe ',data,' ',meanp,' ',psip,' ',params,' ',NITER,' ',BURNIN,' ',NSAMPLE];
fprintf(1,[cmd , '\n']);
system(cmd);

%Combine multiple label samples
prediction=readMat([data '.labels']);
likelihoods=readMat([data '.likelihood']);
predlabs = align_labels(prediction');

%Plot
subplot(1,2,2);
scatter(X(:,1),X(:,2),4,predlabs);
title('Estimated');

%F1 scores
evaluationTable(labels,predlabs)
