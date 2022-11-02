%% LSTM Network applied on the whole toyota dataset to directly predict RUL
%% Descriptors are features in cleaned Toyota dataset
%% Loading Data
clc
clear all
close all

a = load("Toyota Clean Dataset\2017-05-12_dati_ripuliti.mat");
b = load("Toyota Clean Dataset\2017-06-30_dati_ripuliti.mat");
c = load("Toyota Clean Dataset\2018-04-12_dati_ripuliti.mat");

%Removing a 0 in front of each capacity vector in first batch
for i=1:numel(a.toyota)
    a.toyota(i).summary.QDischarge(1) = a.toyota(i).summary.QDischarge(2);
    a.toyota(i).summary.QCharge(1) = a.toyota(i).summary.QCharge(2);
    a.toyota(i).summary.IR(1) = a.toyota(i).summary.IR(2);
    a.toyota(i).summary.Tmax(1) = a.toyota(i).summary.Tmax(2);
    a.toyota(i).summary.Tavg(1) = a.toyota(i).summary.Tavg(2);
    a.toyota(i).summary.Tmin(1) = a.toyota(i).summary.Tmin(2);
    a.toyota(i).summary.chargetime(1) = a.toyota(i).summary.chargetime(2);
end

d= [a.toyota b.toyota c.toyota];
numObservation = numel(d);
%%
for i=5:5 %numel(a.toyota)
    figure
    batteria = a.toyota(i).summary;
    plot(batteria.cycle, batteria.QDischarge);
    hold on
    plot(batteria.cycle, batteria.QCharge);
    legend show
end


%% Dataset extraction
clc


X = cell(numObservation, 1);
Y = cell(numObservation,1);
numFeatures = 8;

for i=1:numObservation
    batt = d(i).summary;

    %batt.cycle removed
    X{i} = [batt.cycle batt.QDischarge batt.QCharge batt.IR batt.Tmax batt.Tavg batt.Tmin batt.chargetime];
    %X{i} = d(i).summary.QDischarge;
    X{i} = transpose(X{i});
    %EoL_capacity = max(X{i}(2,:))*0.85;
    

    Y{i} = flip(d(i).summary.cycle);
    Y{i} = transpose(Y{i});
end

%% Normalize features values (0 mean, 1 variance)

mu = mean([X{:}],2);
sig = std([X{:}],0,2);

for i = 1:numel(X)
    X{i} = (X{i} - mu) ./ sig;
end


%% Set max RUL at 1500 in Y. In this way, data with higher RUL will be less important. This improves reducing the peaks in training error at the beginning of each epoch


%{
thr = 800;
for i = 1:numel(Y)
    Y{i}(Y{i} > thr) = thr;
end
%}


%% Sort data by length of the sequences. This will allow to introduce less padding in the sequences belonging to the same batch
%This will introduce a peak in the training error, at the beginning of each
%epoch, due to the fact that all the longest squences comes there, and the
%shortest at the end

for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
X = X(idx);
Y = Y(idx);

figure
bar(sequenceLengths)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

%% Remove longest (first 7) and shortests (last 4) instancies

for i=1:7
    X(1) = [];
    Y(1) = [];
end

for i=1:4
    X(end) = [];
    Y(end) = [];
end

for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

%% Train and Test split

testSize = round(numel(X)*0.1);

idx = randperm(numel(X), testSize);
idx = sort(idx, "ascend");

XTest = cell(testSize, 1);
YTest = cell(testSize, 1);
XTrain = X;
YTrain = Y;

%Extract test from dataset
for i=1:testSize
    XTest{i} = X{idx(i)};
    YTest{i} = Y{idx(i)};
end

%{
idx = sort(idx, "descend");
%Remove test instancies from training ones
for i=1:testSize
    XTrain(idx(i)) = [];
    YTrain(idx(i)) = [];
end
%}
%% Validation set 2
validationSize = round((numel(XTrain))*0.2);

rem_idx = setdiff(1:numel(X), idx);

idxv = randperm(numel(rem_idx), validationSize);
idxv = rem_idx(idxv);
idxv = sort(idxv, "ascend");

XValid = cell(validationSize, 1);
YValid = cell(validationSize, 1);

%Extract validation from training
for i=1:validationSize
    XValid{i} = X{idxv(i)};
    YValid{i} = Y{idxv(i)};
end

remove = sort([idx idxv], "descend");
%Remove test instancies from training ones
for i=1:numel(remove)
    XTrain(remove(i)) = [];
    YTrain(remove(i)) = [];
end

%% Plot train+test+validation
idxtr = setdiff(1:numel(sequenceLengths), [idx idxv]);

figure
hold on

for i=1:numel(X)
    if ismember(i, idxtr)
        bar(i, sequenceLengths(i), 'b')
        legend("Train Data");
    else if ismember(i, idxv)
        bar(i, sequenceLengths(i), 'c')
    else bar(i, sequenceLengths(i), 'y')
    end
    end
end
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
legend('Location','northeast')

%% Network


numResponses = size(Y{1},1);
numHiddenUnits = 20;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(300)
    dropoutLayer(0.4)
    fullyConnectedLayer(300)
    dropoutLayer(0.4)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%% Training options

maxEpochs = 700;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'Verbose',true,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency',10, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',600,...
    'LearnRateDropFactor',0.7,...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu', ....
    'Verbose',0);

% Training
net = trainNetwork(XTrain,YTrain,layers,options);

%load("D:\Dataset Toyota\Original Dataset\2017-05-12_batchdata_updated_struct_errorcorrect.mat");

%% Test Predictions

YPred = predict(net,XTest,'MiniBatchSize',1);


%%
%{
for i= 1:numel(YTest)
 figure()
 hold on
 plot(YTest{i},'--')

 plot(YPred{i},'.-')
end
%}

    
%% Visualize predictions

%idx = randperm(numel(YPred),4);
idx = [1];
idx = 1:numel(YTest);
figure
for i = idx%1:numel(idx)
    subplot(4,4,i)
    
    plot(YTest{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([0 2000 + 25])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("RUL")
end
legend(["Test Data" "Predicted"],'Location','southeast')

%% RMSE of predictions

for i = 1:numel(YTest)
    YTestLast(i) = YTest{i}(end);
    YPredLast(i) = YPred{i}(end);
end
figure
rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(YPredLast - YTestLast)
title("RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")

%%
avgRUL = 0;
for i=1:numel(Y)
    avgRUL = avgRUL + mean(Y{i});
end

avgRUL = avgRUL / numel(Y);

%%
rmse / avgRUL 
