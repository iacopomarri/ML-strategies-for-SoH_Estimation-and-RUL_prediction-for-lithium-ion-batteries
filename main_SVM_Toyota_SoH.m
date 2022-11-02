%% SoH estiation on MIT dataset using partial charging curve
%close all
%clear all
clearvars  -except d
clc
%rng(2) 

rng("default")

%%
load("MIT_PCC_Features.mat");
Y = Y_SoH;

X2 = load("MIT_features.mat").X;
X3 = load("Partial_MIT_features.mat").X;
%X2 = X2.X;
numObservation = numel(X);

if exist('d','var')==0
    d = MITLoadingCode();
    d(51).policy_readable = strcat(d(51).policy_readable, 'C');
end

%Remove newstring from policies
for i = 1:124
    newPol = policy(i);
    if contains(newPol,"new")
        newPol = extractBetween(newPol, "","C-");
        newPol = strcat(newPol, "C");
        policy{i}= cellstr(newPol);
    end
end

%% Add Variance, min DeltaQ and Temp. to input features
for i=1:numObservation
    X{i}(:,16) = X2{i}(:,1);
    X{i}(:,17) = X2{i}(:,2);
    X{i}(:,18) = X2{i}(:,3);
end

clear X2;
clear Y_SoH;
clear Y_RUL;


%% Add Partial Discharge Variance, min DeltaQ and Temp. to input features
for i=1:numObservation
    X{i}(:,19) = X3{i}(1,:)';
    X{i}(:,20) = X3{i}(2,:)';
    X{i}(:,21) = X3{i}(3,:)';
end

clear X3;
%% Add avg current and max current to features

% for i=1:numObservation
%     pol = d(i).policy_readable;
%     
%     firstC = extractBetween(pol,"","C");
%     firstC = str2double(firstC{1});
%     
%     switch_perc = extractBetween(pol,"(","%");
%     switch_perc = str2double(switch_perc{1});
%     
%     secondC = extractBetween(pol,"-","C");
%     secondC = str2double(secondC{1});
%     
%     normalized_switch_perc = switch_perc /80;
%     
%     feature = firstC*normalized_switch_perc + secondC*(1-normalized_switch_perc);
% 
%     X{i}(:,19) = feature;
% end

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

%% Visualize features over number of cycle

a=range_start:step:range_end;
b=range_start:2*step:range_end;

for i= [10] %20:20:size(X)
%     figure()
%     hold on
%     title ('Features PCC         N° ' + string(i) +  '       Length: ' + string(size(X{i},1)+"        Policy: "+policy{i}), 'FontSize', 15);
%     xlabel('N° of cycle' );
%     ylabel('Charge time' );
%     plot(X{i}(:,1:10))
%     legend(string(a) + ' - ' + string(a+step) + ' V', 'location', 'best','FontSize',10);
% 
%     figure()
%     hold on
%     title ('Features PCC         N° ' + string(i) +  '       Length: ' + string(size(X{i},1)+"        Policy: "+policy{i}), 'FontSize', 15);
%     xlabel('N° of cycle' );
%     ylabel('Charge time' );
%     plot(X{i}(:,11:15))
%     legend(string(b) + ' - ' + string(b+2*step) + ' V', 'location', 'best','FontSize',10);

  
    figure()
    hold on
    %title ('Feature var DeltaQ        N° ' + string(i) +  '       Length: ' + string(size(X{i},1)+"        Policy: "+policy{i}), 'FontSize', 15);
    xlabel('N° of cycle', 'FontSize',20 );
    ylabel('Min Delta Q(V)', 'FontSize',20);
    plot(X{i}(:,20), "LineWidth",1.5)  
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



%% Find a long sequence in the train and sobstitute into thet est

% for i=1:numel(Xtrain)
%     if size(Xtrain{i},1) > 2000
%         i
%         break;
%     end
% end
% 
% temp1 = Xtest{10};
% temp2 = Xtrain{i};
% 
% Xtrain{i} = temp1;
% Xtest{10} = temp2;
% 
% temp1 = Ytest{10};
% temp2 = Ytrain{i};
% 
% Ytrain{i} = temp1;
% Ytest{10} = temp2;


%% Validation set 2
% validationSize = round((numel(Xtrain))*0.00);
% 
% rem_idx = setdiff(1:numel(X), idx);
% 
% idxv = randperm(numel(rem_idx), validationSize);
% idxv = rem_idx(idxv);
% idxv = sort(idxv, "ascend");
% 
% Xvalid = cell(validationSize, 1);
% Yvalid = cell(validationSize, 1);
% 
% %Extract validation from training
% for i=1:validationSize
%     Xvalid{i} = X{idxv(i)};
%     Yvalid{i} = Y{idxv(i)};
% end
% 
% remove = sort([idx idxv], "descend");
% %Remove test instancies from training ones
% for i=1:numel(remove)
%     Xtrain(remove(i)) = [];
%     Ytrain(remove(i)) = [];
% end

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

ftr_idx = [20 21]; %[16 18];
hyperpar = ["BoxConstraint", "KernelScale", "Epsilon"];

model = fitrsvm(X_array_train(:,ftr_idx),Y_array_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true);
%cross_model = fitrsvm(XTrain_array(:,ftr_idx),YTrain_array, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true, KernelScale=0.6, KFold=5);



%% Cross Validation on sequence format data
% 
% K = 5;
% Cross_valid_R2 = [];
% validation_size = floor(size(Xtrain,1)/K);
% 
% for i=1:K
%     disp("running for K = " + int2str(i) + " ...");
%     
%     X_crossval_train = cell(size(Xtrain,1), 1);
%     Y_crossval_train = cell(size(Xtrain,1), 1);
%     X_crossval_valid = [];
%     Y_crossval_valid = [];
% 
%     % copy the full train set
%     for j=1:size(Xtrain,1)
%         X_crossval_train{j,1} = Xtrain{j}(:, ftr_idx);
%         Y_crossval_train{j,1} = Ytrain{j};
%     end
% 
%     % pick validation samples from train set
%     for j=validation_size*(i-1)+1:validation_size*i
%         X_crossval_valid = vertcat(X_crossval_valid, X_crossval_train{j});
%         Y_crossval_valid = [Y_crossval_valid Y_crossval_train{j}];
%     end
%  
%     % delete validation samples from train set
%     X_crossval_train(validation_size*(i-1)+1:validation_size*i) = [];
%     Y_crossval_train(validation_size*(i-1)+1:validation_size*i) = [];
% 
%     % transform remaining train data into array format
%     Xtemp = [];
%     Ytemp = [];
%     for j=1:size(X_crossval_train,1)
%         Xtemp = vertcat(Xtemp, X_crossval_train{j});
%         Ytemp = [Ytemp Y_crossval_train{j}];
%     end
% 
%     X_crossval_train = Xtemp;
%     Y_crossval_train = Ytemp;
%     clear Xtemp Ytemp;
%  
%      cross_model = fitrsvm(X_crossval_train, Y_crossval_train,...
%          BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true);
%  
%      Cross_valid_R2(i) = loss(cross_model,X_crossval_valid, Y_crossval_valid, 'LossFun', @Rsquared);
% end
% 
% mean(Cross_valid_R2)


%% Cross Validation on array format data


% K = 5;
% 
% %metric array
% Cross_valid_R2 = [];
% validation_size = size(X_array_train,1)/K;
% 
% for i=1:K
%     disp("running for K = " + int2str(i) + " ...");
%     
%     %selecting the needed features
%     X_crossval_train = X_array_train(: , ftr_idx);
%     Y_crossval_train = Y_array_train;
%     
%     %picking the Kth part of training for validation
%     X_crossval_valid = X_crossval_train(validation_size*(i-1)+1:validation_size*i, :);
%     Y_crossval_valid = Y_crossval_train(validation_size*(i-1)+1:validation_size*i);
% 
%     %removing validation data from training
%     X_crossval_train(validation_size*(i-1)+1:validation_size*i, :) = [];
%     Y_crossval_train(validation_size*(i-1)+1:validation_size*i) = [];
% 
%     %model
%     cross_model = fitrsvm(X_crossval_train, Y_crossval_train,...
%         BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true);
% 
%     %error
%     Cross_valid_R2(i) = loss(cross_model,X_crossval_valid, Y_crossval_valid, 'LossFun', @Rsquared);
% end
% 
% mean(Cross_valid_R2)




%% Cross validation R2
%Cross_validated_R2 = kfoldLoss(cross_model,'LossFun', @r_squared, 'Mode','individual')

%% Estimate test samples 
Test_R2 = [];
%Xtest = Xtrain;
%Ytest = Ytrain;

for i=1:size(Xtest,1)
    figure()
    hold on;
    legend('location', 'best','FontSize',16);
    xlabel('# cycle','FontSize',20 );
    ylabel('SoH','FontSize',20 );
    %title ('Features PCC         N° ' + string(i) +  '       Length: ' + string(size(X{i},1)+"        Policy: "+policy{i}), 'FontSize', 15);
    title ('Test sample '+string(i)+  '       Life cycles: ' + string(size(Xtest{i},1))+"        Policy: "+policy{i},'FontSize',15 );

    Test_R2(i) = loss(model,Xtest{i}(:,ftr_idx), Ytest{i}, 'LossFun', @Rsquared);
   
%     %R2_train = loss(model, XTrain(:,[7 15]), YTrain, 'LossFun', @Rsquared);
    pred = predict(model,Xtest{i}(:,ftr_idx));
    plot([1:length(Ytest{i})], Ytest{i},  'color','#115eed', 'LineWidth',2,'DisplayName', "Measured "); %+ num2str(R2_train));
    plot ([1:length(Ytest{i})], pred,'.-', 'color','#e87917', 'DisplayName', "Estimated.  R^2: "+ num2str(Test_R2(i)));
end

mean(Test_R2)

%%

figure
%rmse = sqrt(mean((YPredLast - YTestLast).^2))
histogram(Test_R2)
title("Mean R^2 = " + mean(Test_R2))
ylabel("Frequency",'FontSize',20 )
xlabel("R^2 of estimated test samples",'FontSize',20 )



%% PCA analysis on all 3 batteries data

%PCA
figure()
a = [X_array(:,ftr_idx) Y_array'];
[coeffs,score,~,~,expl] = pca(a);
pareto(expl);

