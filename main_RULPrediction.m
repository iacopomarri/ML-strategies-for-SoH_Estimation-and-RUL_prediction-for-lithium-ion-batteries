%% RUL prediction experiments
% Note that this is just a mild, raw first attempt, to set up the RUL
% prediction problem

clear all
clc
load('B0005.mat')
load('B0006.mat')
load('B0007.mat')
load('B0018.mat')

%% Measure nominal capacities
nom_capacity5 = B0005.cycle(2).data.Capacity;
nom_capacity6 = B0006.cycle(2).data.Capacity;
nom_capacity7 = B0007.cycle(2).data.Capacity;
nom_capacity18 = B0018.cycle(3).data.Capacity;

%% Extract the features dataset. 

% (we keep only Y, and will use as feature X the number of cycle (1 to lenght(Y))
[~ , Y] = ExtractPartialCurve(B0005,3.6,0.2,4);
X = 1:length(Y);

%replace Capacity with SoH
Y = Y/nom_capacity5;
%% Train and Test

%This sets the cycle from which we predict
training_split = 0.6;
trainset_length = cast((length(X) * training_split), 'uint8');

X_train = X(1:trainset_length);
Y_train = Y(1:trainset_length);
X_test = X(trainset_length+1:end);
Y_test = Y(trainset_length+1:end);
%% Model and error
model = fitrsvm(transpose(X_train), Y_train, 'Standardize', true); %, 'KernelFunction', 'polynomial');
pred = predict(model, transpose(X_test));
full_pred = predict(model, transpose(X));
err = loss(model,transpose(X_test), Y_test);

%% Plots
figure();
plot(X, Y,'r');
hold on;
scatter (trainset_length+1:length(Y),pred, 'x','b');
xline(double(trainset_length), '-', 'color','#c2ad5d', 'LineWidth',1.5);
%scatter (1:length(Y),full_pred,'green')
xlabel('N° of cycle') 
ylabel('SoH')
title ('SVM trained on B0005 for RUL prediction, with N° of cycle as only input, linear kernel')
error_text = ["MSE: " num2str(err)]
subtitle(error_text)
%%
