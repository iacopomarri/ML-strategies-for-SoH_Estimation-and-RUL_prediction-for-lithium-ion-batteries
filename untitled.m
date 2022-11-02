%% LSTM Network applied on the whole toyota dataset to directly predict RUL
%% Descriptors are features by MIT paper 
%%  Load MIT feature (variance of DeltaQ curves, min of DeltaQ curves, integral temperature)    
clc 
%clear all
clearvars -except d net
close all

rng("default")

load("MIT_Disch_Features.mat")
load("../../RUL features tries/3_3.4 V/Partial_MIT_features_3 to 3,4.mat");
%d = MITLoadingCode();

ftr_idx = [2 3 4]; %variance deltaQ, integral temp, num cycle
numFeatures = numel(ftr_idx);

numObservation = numel(X);

%% Adding N. of cycle as 4th feature

for i=1:numObservation
    X{i}(:,4) = flip(Y_RUL{i});
end

%% pick selected features

for i=1:numObservation
    X{i} = X{i}(:,ftr_idx);
end

%% Transpose dataset to then normalize it
for i=1:numObservation
    X{i} = X{i}';
end
%% Normalize features values (0 mean, 1 variance)

mu = mean([X{:}],2);

sig = std([X{:}],0,2);

for i = 1:numel(X)
    X{i} = (X{i} - mu) ./ sig;
end

%% Transpose dataset back to normal
% for i=1:numObservation
%     X{i} = X{i}';
% end

%% Set max RUL at 1500 in Y. In this way, data with higher RUL will be less important. This improves reducing the peaks in training error at the beginning of each epoch


thr = 900;
for i = 1:numel(Y_RUL)
    Y_RUL{i}(Y_RUL{i} > thr) = thr;
end

Y = Y_RUL;
clear Y_SoH Y_RUL;

%% Sort data by length of the sequences. This will allow to introduce less padding in the sequences belonging to the same batch
%This will introduce a peak in the training error, at the beginning of each
%epoch, due to the fact that all the longest squences comes there, and the
%shortest at the end

sequenceLengths = [];
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

%% Remove longest (first 3) and shortests (last 2) instancies

for i=1:3
    X(1) = [];
    Y(1) = [];
end

for i=1:2
    X(end) = [];
    Y(end) = [];
end

for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

numObservation = numel(X);

%% Train and Test split

testSize = round(numel(X)*0.08);

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

%% Validation set 2
% validationSize = round((numel(XTrain))*0.15);
% 
% rem_idx = setdiff(1:numel(X), idx);
% 
% idxv = randperm(numel(rem_idx), validationSize);
% idxv = rem_idx(idxv);
% idxv = sort(idxv, "ascend");
% 
% XValid = cell(validationSize, 1);
% YValid = cell(validationSize, 1);
% 
% %Extract validation from training
% for i=1:validationSize
%     XValid{i} = X{idxv(i)};
%     YValid{i} = Y{idxv(i)};
%  end
% 
% remove = sort([idx idxv], "descend");
% %Remove test instancies from training ones
% for i=1:numel(remove)
%     XTrain(remove(i)) = [];
%     YTrain(remove(i)) = [];
% end

%% Find a long sequence in the train and sobstitute into the test

for i=1:numel(XTrain)
    if size(XTrain{i},2) > 1200 & size(XTrain{i},2) < 1450
        i
        break;
    end
end

temp1 = XTest{1};
temp2 = XTrain{i};

XTrain{i} = temp1;
XTest{1} = temp2;

temp1 = YTest{1};
temp2 = YTrain{i};

YTrain{i} = temp1;
YTest{1} = temp2;

%move the 1850 from train to test
XTest{end+1} = XTrain{1};
YTest{end+1} = YTrain{1};

XTrain(1) = [];
YTrain(1) = [];

%move the 491 from test to train
XTrain{end+1} = XTest{8};
YTrain{end+1} = YTest{8};

XTest(8)=[];
YTest(8)=[];


%order sequencies
sequenceLengths = [];
for i=1:numel(XTrain)
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
XTrain = XTrain(idx);
YTrain = YTrain(idx);



%%
%order test sequencies
sequenceLengths = [];
for i=1:numel(XTest)
    sequence = XTest{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
XTest = XTest(idx);
YTest = YTest(idx);

% XTest{end+1} = X{end};
% YTest{end+1} = Y{end};

%% Plot train+test+validation
% idxtr = setdiff(1:numel(X), [idx idxv]);
% 
% figure
% hold on
% 
% for i=1:numel(X)
%     if ismember(i, idxtr)
%         bar(i, sequenceLengths(i), 'b')
%         legend("Train Data");
%     else if ismember(i, idxv)
%         bar(i, sequenceLengths(i), 'c')
%     else bar(i, sequenceLengths(i), 'y')
%     end
%     end
% end
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")
% legend('Location','northeast')



%% Network

numResponses = size(Y{1},1);
numHiddenUnits = 80;


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

maxEpochs = 1200;
miniBatchSize = 8;
crossval = false;

options = trainingOptions('adam', ...
    'Verbose',true,...   
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationFrequency',20, ...
    'ValidationPatience',350,...
    'InitialLearnRate', 0.00225, ... %0.00225, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',300,...  %200
    'LearnRateDropFactor',0.5,... %0.25
    'L2Regularization', 0.0005,...
    'GradientThreshold',1, ...
    'Shuffle','never', ...  
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu', ....
    'Verbose',0);
    %'ValidationData', {XValid, YValid}, ...
    %'OutputNetwork','best-validation-loss',...


%% Cross Validation on sequence format data

if crossval
    K = 5;
    Cross_valid_RMSE = [];
    numObservationTrain = numel(XTrain);
    cval_size = floor(size(XTrain,1)/K);
    
    %shuffle dataset to pick randomized validation sets. re-order training set after
    shuffle = randperm(numObservationTrain);
    XTrain = XTrain(shuffle);
    YTrain = YTrain(shuffle);
    
    for i=1:K
        disp("running for K = " + int2str(i) + " ...");
        
        X_crossval_train = cell(size(XTrain,1), 1);
        Y_crossval_train = cell(size(XTrain,1), 1);
        X_crossval_valid = cell(0, 1);
        Y_crossval_valid = cell(0, 1);
    
        % copy the full train set
        for j=1:size(XTrain,1)
            X_crossval_train{j,1} = XTrain{j};
            Y_crossval_train{j,1} = YTrain{j};
        end
    
        % pick validation samples from train set
        for j=cval_size*(i-1)+1:cval_size*i
            X_crossval_valid{end+1} = X_crossval_train{j};
            Y_crossval_valid{end+1} = Y_crossval_train{j};
        end
    
        X_crossval_valid = X_crossval_valid';
        Y_crossval_valid = Y_crossval_valid';
     
        % delete validation samples from train set
        X_crossval_train(cval_size*(i-1)+1:cval_size*i) = [];
        Y_crossval_train(cval_size*(i-1)+1:cval_size*i) = [];
    
        % order by length the training sqeuncies
        sequenceLengths = [];
        for j=1:numel(X_crossval_train)
            sequence = X_crossval_train{j};
            sequenceLengths(j) = size(sequence,2);
        end    
        [sequenceLengths,idx] = sort(sequenceLengths,'descend');
        X_crossval_train = X_crossval_train(idx);
        Y_crossval_train = Y_crossval_train(idx);
     
         
         %set validation options for training
         options.ValidationData = {X_crossval_valid, Y_crossval_valid};
         options.OutputNetwork = 'best-validation-loss';
    
         %train and predict
         cross_model = trainNetwork(X_crossval_train,Y_crossval_train,layers,options);
         YPred = predict(cross_model,X_crossval_valid,'MiniBatchSize',1);
    
         rmse = [];
         for j=1:size(X_crossval_valid,1)
            rmse(j) = sqrt(mean((YPred{j} - Y_crossval_valid{j}).^2));
         end
    
         Cross_valid_RMSE(i) = mean(rmse);
    end
    
    mean(Cross_valid_RMSE)
end



%% validation split

validationSize = round(numel(XTrain)*0.05);

idx = randperm(numel(XTrain), testSize);
idx = sort(idx, "ascend");

XValid = cell(validationSize, 1);
YValid = cell(validationSize, 1);

%Extract validation samples
for i=1:validationSize
    XValid{i} = XTrain{idx(i)};
    YValid{i} = YTrain{idx(i)};
end


idx = sort(idx, "descend");
%Remove test instancies from training ones

for i=1:validationSize
    XTrain(idx(i)) = [];
    YTrain(idx(i)) = [];
end

options.ValidationData = {XValid, YValid};
options.OutputNetwork = 'best-validation-loss';




%% Train final model

%set validation options for training

%options.MaxEpochs = 800;

net = trainNetwork(XTrain,YTrain,layers,options);


%% Test Predictions

YPred = predict(net,XTest,'MiniBatchSize',1);

%% Predict test samples

%Xtest = Xtrain;
%Ytest = Ytrain;
rmse = [];
error = [];

for i=1:size(XTest,1)

    rmse(i) = sqrt(mean((YPred{i} - YTest{i}).^2));
    error(i) = mean(YPred{i} - YTest{i});
   
    figure()
    hold on;
     ylim([0 size(YTest{i},2) + 25])
    %legend('location', 'best','FontSize',12);
    xlabel('N° of cycle','FontSize',18 );
    ylabel('RUL','FontSize',18 );
    %title ('Features PCC         N° ' + string(i) +  '       Length: ' + string(size(X{i},1)+"        Policy: "+policy{i}), 'FontSize', 15);
    title ('Test sample '+string(i)+  '    Life cycles: ' + string(size(YTest{i},2)) +'    RMSE: '+string(rmse(i)), 'FontSize',18 ); 
    diff = size(YTest{i},2)-800;
    
    %shadedplot(1:size(YTest{i},2), YTest{i}+40, YTest{i}-40, [0.7 0.7 1],'#7ca9fc');
    if diff>0
         shadedplot(diff+1:diff+1+800, YTest{i}(end-800:end)+40, YTest{i}(end-800:end)-40, [0.7 0.7 1],'#7ca9fc');
    else
        shadedplot(1:size(YTest{i},2), YTest{i}+40, YTest{i}-40, [0.7 0.7 1],'#7ca9fc');
    end

    hold on
    
     plot(flip(1:size(YTest{i},2)), 'color','#115eed', 'LineWidth',2, 'DisplayName', "Measured");
  %  plot(YTest{i}, 'color','#115eed', 'LineWidth',2, 'DisplayName', "Measured");

   if diff>0
         plot (diff+1:diff+1+800, YPred{i}(end-800:end),'.-', 'color','#e87917', 'DisplayName', "Predicted,  Test RMSE: "+ num2str(rmse(i)));
   else
         plot(YPred{i},'.-', 'color','#e87917', 'DisplayName', "Predicted,  Test RMSE: "+ num2str(rmse(i)));
   end
   
    if diff>0
         xline(diff, '--', 'Color','r', 'LineWidth',1.5 ); %, 'DisplayName', "K = "+ num2str(120));
    end
    legend({'','','± 40', '','Measured','Predicted', 'Start of prediction'},'location', 'best','FontSize',12)
end

mean(rmse)


    
%% Visualize predictions on multiplot window

% %idx = randperm(numel(YPred),4);
idx = 1:numel(YTest);
figure
for i = idx%1:numel(idx)
    subplot(4,4,i)
    
    plot(YTest{idx(i)},'--')
    hold on
    plot(YPred{idx(i)},'.-')
    hold off
    
    ylim([0 900 + 25])
    title("Test Observation " + idx(i))
    xlabel("Time Step")
    ylabel("RUL")
end
legend(["Test Data" "Predicted"],'Location','southeast')

%% RMSE of predictions
  
for i = 1:numel(YTest)
    
end

figure
%rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(error)
title("Mean RMSE = " + mean(rmse))
ylabel("Frequency")
xlabel("Error: Predicted - Test")