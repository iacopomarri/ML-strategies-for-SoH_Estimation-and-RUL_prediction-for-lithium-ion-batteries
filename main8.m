%% Same SVM adding total time feature and number of cycle feature
%% Setting up datasets
close all
clear all
clc

load('B0005.mat');
load('B0006.mat');
load('B0007.mat');
load('B0018.mat');

%% Extracting full cycle time feature
times5 = [];
times6 = [];
times7 = [];

[ch,dis, ~] = ExtractCyclesIndices(B0005);
for i=1:length(ch)
    times5(i) = B0005.cycle(ch(i)).data.Time(end) + B0005.cycle(dis(i)).data.Time(end);
end


[ch,dis, ~] = ExtractCyclesIndices(B0006);
for i=1:length(ch)
    times6(i) = B0006.cycle(ch(i)).data.Time(end) + B0006.cycle(dis(i)).data.Time(end);
end


[ch,dis, imp] = ExtractCyclesIndices(B0007);
for i=1:length(ch)
    times7(i) = B0007.cycle(ch(i)).data.Time(end) + B0007.cycle(dis(i)).data.Time(end);
end

%% Build datasets
% Measure nominal capacities
nom_capacity5 = B0005.cycle(2).data.Capacity;
nom_capacity6 = B0006.cycle(2).data.Capacity;
nom_capacity7 = B0007.cycle(2).data.Capacity;
nom_capacity18 = B0018.cycle(3).data.Capacity;

%Build datasets from batteries
start_range1 = 3.7;
end_range1 = 3.9;
step = 0.05;

[X_5, Y_5] = ExtractPartialCurve(B0005,start_range1,step,end_range1);
[X_6, Y_6] = ExtractPartialCurve(B0006,start_range1,step,end_range1);
[X_7, Y_7] = ExtractPartialCurve(B0007,start_range1,step,end_range1);
[X_18, Y_18] = ExtractPartialCurve(B0018,start_range1,step,end_range1);

Y_5 = Y_5/nom_capacity5;
Y_6 = Y_6/nom_capacity6;
Y_7 = Y_7/nom_capacity7;
Y_18 = Y_18/nom_capacity18;

%Total cycle time feature
%{
X_5(:,end+1) = times5;
X_6(:,end+1) = times6;
X_7(:,end+1) = times7;
%}
%Cycle number feature

X_5(:,end+1) = 1:length(X_5);
X_6(:,end+1) = 1:length(X_6);
X_7(:,end+1) = 1:length(X_7);


%%

%% Compute and plot results
figure()
hold on;
legend('location', 'best');
xlabel('N° of cycle') ;
ylabel('SoH');
title ('SVM LOOCV error on B0005/6/7');

BC =  0.1989;      % >0.2 ok
KS = 11.55;   
Eps = 0.001238;    


%Compute the test error performing a LOOCV between all batteries we have
X_train = vertcat(X_6, X_7);
Y_train = [Y_6 Y_7];
X_test = X_5;
Y_test = Y_5;
%Fit model, compute prediction and R2
%model_a = fitrsvm(X_train,Y_train, BoxConstraint = model_1.ModelParameters.BoxConstraint, Epsilon=model_1.ModelParameters.Epsilon, KernelScale=model_1.ModelParameters.KernelScale);
model_a = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
R2a = loss(model_a,X_test, Y_test, 'LossFun', @Rsquared);
R2a_train = loss(model_a, X_train, Y_train, 'LossFun', @Rsquared);
pred_a = predict(model_a,X_test);
plot([1:length(Y_test)], Y_test, 'r', 'DisplayName', "Measured: B5,  Training R^2: "+ num2str(R2a_train));
plot ([1:length(Y_test)], pred_a,'r--', 'DisplayName', "Estimated,  Test R^2: "+ num2str(R2a));

X_train = vertcat(X_5, X_7);
Y_train = [Y_5 Y_7];
X_test = X_6;
Y_test = Y_6;
%Fit model, compute prediction and R2
%model_b = fitrsvm(X_train,Y_train, BoxConstraint = model_1.ModelParameters.BoxConstraint, Epsilon=model_1.ModelParameters.Epsilon, KernelScale=model_1.ModelParameters.KernelScale);
model_b = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
R2b = loss(model_b,X_test, Y_test, 'LossFun', @Rsquared);
R2b_train = loss(model_b, X_train, Y_train, 'LossFun', @Rsquared);
pred_b = predict(model_b,X_test);
plot ([1:length(Y_test)], Y_test,'b', 'DisplayName', "Measured: B6, Training   R^2: "+ num2str(R2b_train));
plot ([1:length(Y_test)], pred_b,'b--', 'DisplayName', "Estimated,  Test  R^2: "+ num2str(R2b));

X_train = vertcat(X_5, X_6);
Y_train = [Y_5 Y_6];
X_test = X_7;
Y_test = Y_7;
%Fit model, compute prediction and R2
%model_c = fitrsvm(X_train,Y_train, BoxConstraint = model_1.ModelParameters.BoxConstraint, Epsilon=model_1.ModelParameters.Epsilon, KernelScale=model_1.ModelParameters.KernelScale);
model_c = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
R2c = loss(model_c,X_test, Y_test, 'LossFun', @Rsquared);
pred_c = predict(model_c,X_test);
R2c_train = loss(model_c, X_train, Y_train, 'LossFun', @Rsquared);
plot ([1:length(Y_test)], Y_test,'g', 'DisplayName', "Measured: B7, Training   R^2: "+ num2str(R2c_train));
plot ([1:length(Y_test)], pred_c,'g--', 'DisplayName', "Estimated, Test R^2: "+ num2str(R2c));

R2 = (R2a+R2b+R2c)/3;

error_text = ["R^2: " num2str(R2)];
subtitle(error_text);


