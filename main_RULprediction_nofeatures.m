clearvars -except d 
clc
close all 

if exist('d','var')==0
    d = MITLoadingCode();
 
end

numObservation = numel(d);
%%
numObservation = 100;
data = cell(numObservation, 1);
dataY = cell(numObservation, 1);

min_times = [];

for k=1:numObservation
    max_times_cycles = [];
    batt = d(k);
    n_cycl = size(batt.cycles,2);
    cycles = cell(n_cycl,6);

    for i=1:n_cycl
        cycl = batt.cycles(i);

        cycles{i,1} = cycl.t;
        cycles{i,2} = cycl.Qc;
        cycles{i,3} = cycl.I;
        cycles{i,4} = cycl.V;
        cycles{i,5} = cycl.T;
        cycles{i,6} = cycl.Qd; 
        
        %find max time t per cycle
        max_times_cycles(i) = max(cycl.t);
    end    
    data{k} =  cycles';
    dataY{k} = flip(1:n_cycl);

    %find the minimum "max time" of the cycles of batt k.
    min_times(k) = min(max_times_cycles);
end

min(min_times)
clear k i dim cycles










%% Network

numResponses = size(Y{1},1);
numHiddenUnits = 5;
numFeatures = 6;


layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(200)
    dropoutLayer(0.4)
    fullyConnectedLayer(50)
    dropoutLayer(0.4)
    fullyConnectedLayer(50)
    dropoutLayer(0.4)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%% Training options

maxEpochs = 1000;
miniBatchSize = 1;

options = trainingOptions('adam', ...
    'Verbose',true,...   
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',300,...  %150
    'LearnRateDropFactor',0.0625,...  %0.25
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu', ....
    'Verbose',0);

% Training
net = trainNetwork(X,Y,layers,options);
