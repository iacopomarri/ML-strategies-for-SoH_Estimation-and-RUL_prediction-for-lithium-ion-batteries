%% SVR Hyperparameters optimization. 3 experiments are made, in order
%SVR optimized and validated on 3 batteries with LOOCV.
%SVR optimized on 2 batteries, validated on the third.
%SVR used for fitting B5 (optimized and validated on B5).
%% Setting up datasets
close all
clearvars  -except d
clc


%% Building feature sets for each battery
load('B0005.mat');
load('B0006.mat');
load('B0007.mat');
load('B0018.mat');

% Measure nominal capacities
nom_capacity5 = B0005.cycle(2).data.Capacity;
nom_capacity6 = B0006.cycle(2).data.Capacity;
nom_capacity7 = B0007.cycle(2).data.Capacity;
nom_capacity18 = B0018.cycle(3).data.Capacity;

%Build datasets from batteries
start_range1 = 3.9; %3.9;
end_range1 = 4; %4;
step = 0.05;%0.1;

%{
X_5 = [X_5 ExtractTotalMovedCharge(B0005)];
X_6 = [X_6 ExtractTotalMovedCharge(B0006)];
X_7 = [X_7 ExtractTotalMovedCharge(B0007)];
X_18 = [X_18 ExtractTotalMovedCharge(B0018)];

X_5 = ExtractTotalMovedCharge(B0005);
X_6 = ExtractTotalMovedCharge(B0006);
X_7 = ExtractTotalMovedCharge(B0007);
X_18 = ExtractTotalMovedCharge(B0018);
%}

[X_5, Y_5] = ExtractPartialCurve(B0005,start_range1,step,end_range1);
[X_6, Y_6] = ExtractPartialCurve(B0006,start_range1,step,end_range1);
[X_7, Y_7] = ExtractPartialCurve(B0007,start_range1,step,end_range1);
[X_18, Y_18] = ExtractPartialCurve(B0018,start_range1,step,end_range1);

Y_5 = Y_5/nom_capacity5;
Y_6 = Y_6/nom_capacity6;
Y_7 = Y_7/nom_capacity7;
Y_18 = Y_18/nom_capacity18;


% seems that adding more than 2 features decreases the performances. If we
% had more than 3 data we coould handle more complexity

% X_5 = [X_5 ExtractTotalMovedCharge(B0005)];
% X_6 = [X_6 ExtractTotalMovedCharge(B0006)];
% X_7 = [X_7 ExtractTotalMovedCharge(B0007)];
% X_18 = [X_18 ExtractTotalMovedCharge(B0018)];










%% Visualize features over number of cycle B5

a=start_range1:step:end_range1;

figure()
hold on
%title ('PCC Voltage steps charge time  -  B0005', 'FontSize', 12);
xlabel('# cycle','FontSize',18 );
ylabel('Charge time','FontSize',18 );
plot(X_5)
legend(string(a) + ' - ' + string(a+step) + ' V', 'location', 'best','FontSize',16);
%cleanfigure('minimumPointsDistance', 0.1)
%matlab2tikz('..\..\thesis\B0005_PCC_Features.tex');

%% Visualize features over number of cycle B6
figure()
hold on
%title ('PCC Voltage steps charge time  -  B0006', 'FontSize', 12);
xlabel('# cycle' ,'FontSize',18 );
ylabel('Charge time','FontSize',18 );
plot(X_6)
legend(string(a) + ' - ' + string(a+step) + ' V', 'location', 'best','FontSize',16);
cleanfigure('minimumPointsDistance', 0.1)
%matlab2tikz('..\..\thesis\B0006_PCC_Features.tex');

%% Visualize features over number of cycle B7
figure()
hold on
%title ('PCC Voltage steps charge time  -  B0007', 'FontSize', 12);
xlabel('# cycle','FontSize',18 );
ylabel('Charge time','FontSize',18 );
plot(X_7)
legend(string(a) + ' - ' + string(a+step) + ' V', 'location', 'best','FontSize',16);
cleanfigure('minimumPointsDistance', 0.1)
%matlab2tikz('..\..\thesis\B0007_PCC_Features.tex');

%% PCA analysis on all data
% clc
% X_train = vertcat(X_5, X_6, X_7);
% Y_train = [Y_5 Y_6 Y_7];
% 
% %PCA
% figure()
% a = [X_train transpose(Y_train)];
% [coeffs,score,~,~,expl] = pca(a);
% pareto(expl);

%% Optimize parameters for all 3 batteries (if fit() is commented is because in the next section, best found parameters are hardcoded and used).
%you may not need to run this section since best parameters are already
%hardcoded
% Models, Results and plots

clc
X_train = vertcat(X_5, X_6, X_7);
Y_train = [Y_5 Y_6 Y_7];



%SVM hyperparams
BC =  0.1989;      % >0.2 ok
KS = 11.55;   
Eps = 0.03013;%0.030107; %0.00128
Kernel = 'linear';

rng default
hyperpar = ["BoxConstraint", "KernelScale", "Epsilon"];
model_1 = fitrsvm(X_train,Y_train,  "OptimizeHyperparameters",hyperpar, "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=100));
BC = 0.001052; %model_1.ModelParameters.BoxConstraint;       
KS = 21.344; %model_1.ModelParameters.KernelScale;
Eps = 0.00128; %model_1.ModelParameters.Epsilon;   
Kernel = 'linear'; % model_1.ModelParameters.KernelFunction;

figure()
hold on;
legend('location', 'best','FontSize',8);
xlabel('N° of cycle','FontSize',18 );
ylabel('SoH','FontSize',18 );
%title ('SVR error on B0005/6/7','FontSize',18 );

%% Compute the test error performing a LOOCV between all batteries we have

X_train = vertcat(X_6, X_7);
Y_train = [Y_6 Y_7];
X_test = X_5;
Y_test = Y_5;
%Fit model, compute prediction and R2
model_a = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel);
R2a = loss(model_a,X_test, Y_test, 'LossFun', @Rsquared);
R2a_train = loss(model_a, X_train, Y_train, 'LossFun', @Rsquared);
pred_a = predict(model_a,X_test);
plot([1:length(Y_test)], Y_test, 'r', 'DisplayName', "Measured B5");
plot ([1:length(Y_test)], pred_a,'r--', 'DisplayName', "Estimated B5: R^2: "+ num2str(R2a));


X_train = vertcat(X_5, X_7);
Y_train = [Y_5 Y_7];
X_test = X_6;
Y_test = Y_6;
%Fit model, compute prediction and R2
model_b = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel);
R2b = loss(model_b,X_test, Y_test, 'LossFun', @Rsquared);
R2b_train = loss(model_b, X_train, Y_train, 'LossFun', @Rsquared);
pred_b = predict(model_b,X_test);
plot ([1:length(Y_test)], Y_test,'b', 'DisplayName', "Measured B6");
plot ([1:length(Y_test)], pred_b,'b--', 'DisplayName', "Estimated B6: R^2: "+ num2str(R2b));


X_train = vertcat(X_5, X_6);
Y_train = [Y_5 Y_6];
X_test = X_7;
Y_test = Y_7;
%Fit model, compute prediction and R2
model_c = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel);
R2c = loss(model_c,X_test, Y_test, 'LossFun', @Rsquared);
pred_c = predict(model_c,X_test);
R2c_train = loss(model_c, X_train, Y_train, 'LossFun', @Rsquared);
plot ([1:length(Y_test)], Y_test,'g', 'DisplayName', "Measured B7");
plot ([1:length(Y_test)], pred_c,'g--', 'DisplayName', "Estimated B7: R^2: "+ num2str(R2c));

R2 = (R2a+R2b+R2c)/3;

error_text = ["Mean R^2: " num2str(R2)];
subtitle(error_text);

% cleanfigure('minimumPointsDistance', 0.1)
% matlab2tikz('..\..\thesis\NASA_results.tex');


% figure()
% hold on
% X_train = vertcat(X_5, X_6, X_7);
% Y_train = [Y_5 Y_6 Y_7];
% X_test = X_18;
% Y_test = Y_18;
% %Fit model, compute prediction and R2
% model_test = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel);
% R2test = loss(model_test,X_test, Y_test, 'LossFun', @Rsquared);
% pred_test = predict(model_test,X_test);
% R2test_train = loss(model_test, X_train, Y_train, 'LossFun', @Rsquared);
% plot ([1:length(Y_test)], Y_test,'g', 'DisplayName', "Measured: B7, Training   R^2: "+ num2str(R2test_train));
% plot ([1:length(Y_test)], pred_test,'g--', 'DisplayName', "Estimated, Test R^2: "+ num2str(R2test));
% 
% R2_text = ["R^2: " num2str(R2test)];
% subtitle(R2_text);

%% Optimize model on B0006/7 to fit B0005
clc

X_train = vertcat(X_6, X_7);


Y_train = [Y_6 Y_7];

X_test = X_5;
Y_test = Y_5;

%SVM hyperparams
BC =  50.045;     % > 0.5 OK  very stable over 100, not so influent
KS = 149.59;      % > 0.1 /10 OK very stable over 100, not so influent
Eps = 0.035;   %0.063 gaussian with TMC   %0.03 linear with TMC   %0.0195 gaussian with PC   %0.065 linear with PC     unstable and influent.  

%BC =   987.86;     % > 0.5 OK
%KS =230.58;      % > 0.1 OK
%Eps = 0.065299;   %0.095   % Molto a caso

rng default
model = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction="polynomial", PolynomialOrder=3);
%model = fitrsvm(X_train,Y_train,  "OptimizeHyperparameters",["Epsilon" "KernelFunction"], "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=100, ShowPlots=false), BoxConstraint = BC, KernelScale=KS);

%Results and plots
R2 = loss(model,X_test, Y_test, 'LossFun', @Rsquared);
pred = predict(model,X_test);   

figure()
hold on;
legend('location', 'best','FontSize',18 );
xlabel('N° of cycle','FontSize',18 );
ylabel('SoH','FontSize',18 );
title ('SVM trained on B0006/7, tested on B0005','FontSize',18 );
plot ([1:length(Y_test)], Y_test,'r-o', 'DisplayName', "Measured");
plot ([1:length(Y_test)], pred,'r--', 'DisplayName', "Estimated,  R^2: "+ num2str(R2));


%% Optimize fitting of B0005 on B0005. No test set
%Is interesting noting that changing the features to eg. 3.8 - 4 V, with
%step 0.01 (AKA adding more and more features and making the model
%more complex) we can overfit B5 and push R2 up over to 0.99+, but losing
%in generalization

clc
X_train = X_5;
Y_train = Y_5;
X_test = X_train;
Y_test = Y_train;

%SVM hyperparams
BC =  247.38;    % > 0.5 OK
KS = 498.86;      % > 0.1 OK
Eps = 0.00177;   %0.095   % Molto a caso

%model
rng default
model = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
%model = fitrsvm(X_train,Y_train,  "OptimizeHyperparameters",'auto', "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=10));

%Results and plots
R2 = loss(model,X_test, Y_test, 'LossFun', @Rsquared);
pred=predict(model, X_test);

figure()
hold on;
legend('location', 'best');
xlabel('N° of cycle') ;
ylabel('SoH');
title ('SVM fitting B0005');
plot ([1:length(Y_test)], Y_test,'r', 'DisplayName', "Measured");
plot ([1:length(Y_test)], pred,'r--', 'DisplayName', "Fitting,  R^2: "+ num2str(R2));
