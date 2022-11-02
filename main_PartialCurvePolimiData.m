    %% Using partial charging curve and tuning SVM on old Polimi data. 
clc
clear all
%close all
%%
load("I_8_20_v2.mat");
load("I_25_02_v2.mat");
load("I_50_v2.mat");
load("B0005.mat");

%% Plotting temp. of each charging cycle
[ch, dis] = ExtractCyclesIndices(I_8_20_v2);
figure()
ylabel('Temperature C°');
hold on
for i=1:1:length(ch)
    plot(I_8_20_v2.cycle(ch(i)).data.Temperature_measured);
    
end
%% Building feature sets for each battery
clc

% Measure nominal capacities
nom_capacity8 = I_8_20_v2.cycle(2).data.Capacity;
nom_capacity25 = I_25_02_v2.cycle(2).data.Capacity;
nom_capacity50 = I_50_v2.cycle(2).data.Capacity;


start_range1 = 3.7;
end_range1 = 3.9;
step = 0.1; %0.01, 0.005, 0.001

%{
X_8 = [X_8 ExtractTotalMovedCharge(I_8_20_v2)];
X_25 = [X_25 ExtractTotalMovedCharge(I_25_02_v2)];
X_50 = [X_50 ExtractTotalMovedCharge(I_50_v2)];

X_8 = ExtractTotalMovedCharge(I_8_20_v2);
X_25 =ExtractTotalMovedCharge(I_25_02_v2);
X_50 = ExtractTotalMovedCharge(I_50_v2);
%}

[X_8,Y_8] = ExtractPartialCurve(I_8_20_v2, start_range1,step,end_range1); 
[X_25, Y_25] = ExtractPartialCurve(I_25_02_v2,start_range1,step,end_range1);
[X_50, Y_50] = ExtractPartialCurve(I_50_v2,start_range1,step,end_range1);

Y_8 = Y_8/nom_capacity8;
Y_25 = Y_25/nom_capacity25;
Y_50 = Y_50/nom_capacity50;

X_8 = [X_8 ExtractTotalMovedCharge(I_8_20_v2)];
X_25 = [X_25 ExtractTotalMovedCharge(I_25_02_v2)];
X_50 = [X_50 ExtractTotalMovedCharge(I_50_v2)];

%% Visualize features over number of cycle

a=start_range1:step:end_range1;

figure()
hold on
title ('Features Batt. 8');
xlabel('N° of cycle' );
ylabel('Charge time' );
plot(X_8)
legend(string(a) + ' - ' + string(a+step) + ' V', 'location', 'best','FontSize',12);

figure()
hold on
title ('Features Batt. 25');
xlabel('N° of cycle' );
ylabel('Charge time' );
plot(X_25)
legend(string(a) + ' - ' + string(a+step) + ' V', 'location', 'best','FontSize',12);

figure()
hold on
title ('Features Batt. 50');
xlabel('N° of cycle' );
ylabel('Charge time' );
plot(X_50)
legend(string(a) + ' - ' + string(a+step) + ' V', 'location', 'best','FontSize',12);


%% PCA analysis on all 3 batteries data

clc
X_train = vertcat(X_25, X_50, X_8);
Y_train = [Y_25 Y_50 Y_8];
X_test = X_50;
Y_test = Y_50;


%PCA
figure()
a = [X_train transpose(Y_train)];
[coeffs,score,~,~,expl] = pca(a);
pareto(expl);
%% Optimize model on I_8/25 to fit I_50
clc
X_train = vertcat(X_25);
Y_train = [Y_25];
X_test = X_50;
Y_test = Y_50;

%SVM hyperparams
BC =  10.045;     % > 0.5 OK
KS = 0.6;      % > 0.1 OK
Eps = 0.000607;    % Molto a caso

%model
rng default
model = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);
%model = fitrsvm(X_train,Y_train,  "OptimizeHyperparameters",'Epsilon', "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=300, ShowPlots=false), BoxConstraint = BC, Epsilon = Eps, KernelScale=KS);

%Results and plots
R2 = loss(model,X_test, Y_test, 'LossFun', @Rsquared);
pred = predict(model,X_test);

figure()
hold on;
legend('location', 'best','FontSize',18 );
xlabel('N° of cycle','FontSize',18 );
ylabel('SoH','FontSize',18 );
title ('SVM trained on I-8/25, tested on I-50','FontSize',18 );
plot ([1:length(Y_test)], Y_test,'r', 'DisplayName', "Measured");
plot ([1:length(Y_test)], pred,'r--', 'DisplayName', "Estimated,  R^2: "+ num2str(R2));