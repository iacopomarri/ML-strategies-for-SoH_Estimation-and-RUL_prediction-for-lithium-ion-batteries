%% Cross Validation for different voltage windows, to find for each, best model, best HP.
close all
clearvars  -except d
clc

rng("default")

%% Load data

% if exist('d','var')==0
%     d = MITLoadingCode();
%     d(51).policy_readable = strcat(d(51).policy_readable, 'C');
% end


load("MIT_PCC_Features.mat"); %Load this only for Y
X = load("../../RUL features tries/2.9_3 V/Partial_MIT_features_2,9 to 3.mat").X;
Y = Y_SoH;

%%

numObservation = numel(X);

%Remove newstring from policies
for i = 1:124
    newPol = policy(i);
    if contains(newPol,"new")
        newPol = extractBetween(newPol, "","C-");
        newPol = strcat(newPol, "C");
        policy{i}= cellstr(newPol);
    end
end
%% Find noisy samples, plot and remove them
idx = [];
for i=1:numel(Y)
    if find(Y{i}>1.1)
        idx = [idx i];
    end
end

% figure()
% hold on
for i=idx
    %lot(Y{i})
    %plot(X{i}(:,[16]))
end

Y(idx) = [];
X(idx) = [];

%recompute num obsv
numObservation = numel(X);
%% Transpose dataset, normalize it, transpose it back

%Transpose
for i=1:numObservation
    X{i} = X{i}';
end

% Normalize features values (0 mean, 1 variance)
mu = mean([X{:}],2);

sig = std([X{:}],0,2);

for i = 1:numel(X)
    X{i} = (X{i} - mu) ./ sig;
end

%Transpose back 
 for i=1:numObservation
     X{i} = X{i}';
 end

%% Train and Test split

%shuffle dataset    
shuffle = randperm(numObservation);
X = X(shuffle);
Y = Y(shuffle);

testSize = round(numObservation*0.08);

idx = randperm(numObservation, testSize);
idx = sort(idx, "ascend");

Xtest = cell(testSize, 1);
Ytest = cell(testSize, 1);
Xtrain = X;
Ytrain = Y;

%Extract test from dataset
for i=1:testSize
    Xtest{i} = X{idx(i)};
    Ytest{i} = Y{idx(i)};
end


idx = sort(idx, "descend");
%Remove test instancies from training ones
for i=1:testSize
    Xtrain(idx(i)) = [];
    Ytrain(idx(i)) = [];
end 

%% Generate dataset in array_format
X_array_train = [];
Y_array_train = [];
X_array = [];
Y_array = [];


for i=1:size(Xtrain)
    X_array_train = vertcat(X_array_train, Xtrain{i});
    Y_array_train = [Y_array_train Ytrain{i}]; %% vertcat(YTrain, Ytrain{i});
end


for i=1:size(X)
    X_array = vertcat(X_array, X{i});
    Y_array = [Y_array Y{i}]; %% vertcat(YTrain, Ytrain{i});
end

%% SVM model

%SVM hyperparams
BC =  0.1; %1.4  %0.1989;      % >0.2 ok
KS = 1; %4.47; %11.55;   
Eps = 0.006078; %  0.03013;%0.030107; %0.00128
Kernel = 'gaussian';
%cell = 85;  

% 1: variance     2: min    3: temperature

hyperpar = ["BoxConstraint", "KernelScale", "Epsilon", "Kernel"];

%model = fitrsvm(X_array_train(:,ftr_idx),Y_array_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true);
%cross_model = fitrsvm(XTrain_array(:,ftr_idx),YTrain_array, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true, KernelScale=0.6, KFold=5);


%% Use Matlab built-in optimizer (Takes too much time)
% init_model = fitrsvm(X_array_train,Y_array_train, "OptimizeHyperparameters", hyperpar, "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=10));
% BC = init_model.ModelParameters.BoxConstraint;
%% Perform crossvalid on 3 models: Var-Temp, Min-Temp, Var-Min-Temp.
model_cvR2 = [];
models = cell(3,1);

models{1} = [1 3];  %variance - temp
models{2} = [2 3]; % min temp
models{3} = [1 2 3];% var min temp
    

model_cvR2(1) = KFoldCV(5, Xtrain, Ytrain, models{1}, BC, Eps, KS, Kernel);
model_cvR2(2) = KFoldCV(5, Xtrain, Ytrain, models{2}, BC, Eps, KS, Kernel);
model_cvR2(3) = KFoldCV(5, Xtrain, Ytrain, models{3}, BC, Eps, KS, Kernel);

[~, index] =  max(model_cvR2);

%% Use the best model to perform CV on many values of the 3 hyperparameters
BC_cvR2 = [];
KS_cvR2 = [];
Eps_cvR2 = [];

multiplcationFactors = [0.2 0.5 1 2 5 10 50 100];

%Values to test
BCs = multiplcationFactors*BC
KSs = multiplcationFactors*KS
Epss = multiplcationFactors*Eps

for i=1:size(multiplcationFactors,2)
    BC_cvR2(i) = KFoldCV(5, Xtrain, Ytrain, models{index}, BCs(i), Eps, KS, Kernel);
    KS_cvR2(i) = KFoldCV(5, Xtrain, Ytrain, models{index}, KSs(i), Eps, KS, Kernel);
    Eps_cvR2(i) = KFoldCV(5, Xtrain, Ytrain, models{index}, Epss(i), Eps, KS, Kernel);
end
%%
%%save("../../RUL features tries/occhioo", "model_cvR2", "BC_cvR2", "KS_cvR2", "Eps_cvR2");

%% plot BC, KS, Eps cross valid R2
figure()
hold on
xlabel("log10 BC value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Box Constraint CV")
plot(log10(BCs), BC_cvR2)

figure()
hold on
xlabel("BC value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Box Constraint CV")
plot(BCs, BC_cvR2)



figure()
hold on
xlabel("KS value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Kernel Scale CV")
plot(KSs, KS_cvR2)

figure()
hold on
xlabel("log10 KS value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Kernel Scale CV")
plot(log10(KSs), KS_cvR2)



figure()
hold on
xlabel("Eps value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Eps CV")
plot(Epss, Eps_cvR2)

figure()
hold on
xlabel("log10 Eps value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Eps CV")
plot(log10(Epss), Eps_cvR2)

%% Select best values for hyperparams

[~, BCindex] =  max(BC_cvR2);
[~, KSindex] =  max(KS_cvR2);
[~, Epsindex] =  max(Eps_cvR2);

BC = BCs(BCindex);
KS = KSs(KSindex);
Eps = Epss(Epsindex);

%% 5fold CV on final model

KFoldCV(5, Xtrain, Ytrain, models{index}, BCs(BCindex), Epss(Epsindex), KSs(KSindex), Kernel)

%% Add 1 more CV for the final optimized model, and save all the 5 R2 resulting


a = [2 3 4 5]
a = [1 a]