%% Same SVM with B0005-6-7-820 LOOCV.
%% USELESS, probabily to delete

%% Setting up datasets
close all
clear all
clc

%%
load('B0005.mat');
load('B0006.mat');
load('B0007.mat');
load('B0018.mat');
B_8_20 = load('I_8_20_v2.mat');

% Measure nominal capacities
nom_capacity5 = B0005.cycle(2).data.Capacity;
nom_capacity6 = B0006.cycle(2).data.Capacity;
nom_capacity7 = B0007.cycle(2).data.Capacity;
nom_capacity18 = B0018.cycle(3).data.Capacity;
nom_capacity820 = B_8_20.cycle(1).data.Capacity;

%Build datasets from batteries
start_range1 = 3.7;
end_range1 = 3.9;
step = 0.05;

[X_5, Y_5] = ExtractPartialCurve(B0005,start_range1,step,end_range1);
[X_6, Y_6] = ExtractPartialCurve(B0006,start_range1,step,end_range1);
[X_7, Y_7] = ExtractPartialCurve(B0007,start_range1,step,end_range1);
[X_18, Y_18] = ExtractPartialCurve(B0018,start_range1,step,end_range1);
[X_820, Y_820] = ExtractPartialCurve(B_8_20,start_range1,step,end_range1);
Y_5 = Y_5/nom_capacity5;
Y_6 = Y_6/nom_capacity6;
Y_7 = Y_7/nom_capacity7;
Y_18 = Y_18/nom_capacity18;
Y_820 = Y_820/nom_capacity820;

%% Compute and plot results
figure()
hold on;
legend('location', 'best');
xlabel('N° of cycle') ;
ylabel('SoH');
title ('SVM LOOCV error on B0005/6/7');

BC =  0.1989      % >0.2 ok
KS = 11.55   
Eps = 0.001238    


%Compute the test error performing a LOOCV between all batteries we have
X_train = vertcat(X_6, X_7, X_820);
Y_train = [Y_6 Y_7 Y_820];
X_test = X_5;
Y_test = Y_5;
%Fit model, compute prediction and R2
%model_a = fitrsvm(X_train,Y_train, BoxConstraint = model_1.ModelParameters.BoxConstraint, Epsilon=model_1.ModelParameters.Epsilon, KernelScale=model_1.ModelParameters.KernelScale);
model_a = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
R2a = loss(model_a,X_test, Y_test, 'LossFun', @Rsquared);
pred_a = predict(model_a,X_test);
plot([1:length(Y_test)], Y_test, 'r', 'DisplayName', 'Measured: B0005');
plot ([1:length(Y_test)], pred_a,'r--', 'DisplayName', "Estimated,  R^2: "+ num2str(R2a));

X_train = vertcat(X_5, X_7, X_820);
Y_train = [Y_5 Y_7 Y_820];
X_test = X_6;
Y_test = Y_6;
%Fit model, compute prediction and R2
%model_b = fitrsvm(X_train,Y_train, BoxConstraint = model_1.ModelParameters.BoxConstraint, Epsilon=model_1.ModelParameters.Epsilon, KernelScale=model_1.ModelParameters.KernelScale);
model_b = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
R2b = loss(model_b,X_test, Y_test, 'LossFun', @Rsquared);
pred_b = predict(model_b,X_test);
plot ([1:length(Y_test)], Y_test,'b', 'DisplayName', "Measured: B0006");
plot ([1:length(Y_test)], pred_b,'b--', 'DisplayName', "Estimated,  R^2: "+ num2str(R2b));

X_train = vertcat(X_5, X_6, X_820);
Y_train = [Y_5 Y_6 Y_820];
X_test = X_7;
Y_test = Y_7;
%Fit model, compute prediction and R2
%model_c = fitrsvm(X_train,Y_train, BoxConstraint = model_1.ModelParameters.BoxConstraint, Epsilon=model_1.ModelParameters.Epsilon, KernelScale=model_1.ModelParameters.KernelScale);
model_c = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
R2c = loss(model_c,X_test, Y_test, 'LossFun', @Rsquared);
pred_c = predict(model_c,X_test);
plot ([1:length(Y_test)], Y_test,'g', 'DisplayName', "Measured: B0007");
plot ([1:length(Y_test)], pred_c,'g--', 'DisplayName', "Estimated,  R^2: "+ num2str(R2c));

X_train = vertcat(X_5, X_6, X_7);
Y_train = [Y_5 Y_6 Y_7];
X_test = X_820;
Y_test = Y_820;
%Fit model, compute prediction and R2
%model_c = fitrsvm(X_train,Y_train, BoxConstraint = model_1.ModelParameters.BoxConstraint, Epsilon=model_1.ModelParameters.Epsilon, KernelScale=model_1.ModelParameters.KernelScale);
model_d = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
R2d = loss(model_c,X_test, Y_test, 'LossFun', @Rsquared);
pred_d = predict(model_c,X_test);
plot ([1:length(Y_test)], Y_test, 'Color','#eb17dd', 'DisplayName', "Measured: B0820");
plot ([1:length(Y_test)], pred_d, '--', 'Color','#eb17dd', 'DisplayName', "Estimated,  R^2: "+ num2str(R2d));

R2 = (R2a+R2b+R2c+R2d)/4;

error_text = ["R^2: " num2str(R2)];
subtitle(error_text);