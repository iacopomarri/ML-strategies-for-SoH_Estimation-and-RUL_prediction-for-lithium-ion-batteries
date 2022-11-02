%% LSTM Network applied on the whole toyota dataset to predict SoH

%% Loading Data
clc
clear all

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

%% Dataset extraction
d= [a.toyota b.toyota c.toyota];
numObservation = numel(d);
splits = [0.6 0.7 0.8];

X = cell(1, 1);
Y = cell(1, 1);

X(1) = [];
Y(1) = [];

for i=1:numObservation
    batt = d(i).summary;

    for j=1:numel(splits)   
        trainSize = round((numel(batt.cycle))*splits(j));
        

        X{end+1} = [batt.cycle(1:trainSize) batt.QDischarge(1:trainSize) batt.QCharge(1:trainSize) batt.IR(1:trainSize) batt.Tmax(1:trainSize) batt.Tavg(1:trainSize) batt.Tmin(1:trainSize) batt.chargetime(1:trainSize)];
        X{end} = transpose(X{end});
    
        Y{end+1} = batt.QDischarge(trainSize+1:end);
        Y{end} = transpose(Y{end});
    end
end

%% //TODO: Normalize features


%% Sort data by length of the sequences. This will allow to introduce less padding in the sequences belonging to the same batch
%This will introduce a peak in the training error, at the beginning of each
%epoch, due to the fact that all the longest squences come there, and the
%shortest at the end

for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
    sequence = Y{i};
    sequenceLengthsY(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
X = X(idx);
Y = Y(idx);

for i=1:numel(Y)
    sequence = Y{i};
    sequenceLengthsY(i) = size(sequence,2);
end

figure()
bar(sequenceLengths)
hold on
bar(sequenceLengthsY)
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

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

idx = sort(idx, "descend");
%Remove test instancies from training ones
for i=1:testSize
    XTrain(idx(i)) = [];
    YTrain(idx(i)) = [];
end


%% Network

numFeatures = 8;
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

%% Network 2

layers = [
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(100)
    dropoutLayer(0.4)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%% Training options

maxEpochs = 400;
miniBatchSize = 20;

% 'ValidationData', {XValid, YValid}, ...
options = trainingOptions('adam', ...
    'Verbose',true,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationFrequency',10, ...
    'InitialLearnRate',0.001, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu', ....
    'Verbose',0,...
     SequencePaddingDirection="left");

% Training
net = trainNetwork(XTrain,YTrain,layers,options);


%% Test Predictions

YPred = predict(net,XTest,'MiniBatchSize',1);


    
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
    
    %ylim([0 thr + 25])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("SoH")
end
legend(["Test Data" "Predicted"],'Location','southeast')