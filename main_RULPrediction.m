%% Here, various functions are used to fit SoH data up to different prediction cycles. 80-100-120
clear all
clc
load('B0005.mat')
load('B0006.mat')
load('B0007.mat')
load('B0018.mat')

% Measure nominal capacities
nom_capacity5 = B0005.cycle(2).data.Capacity;
nom_capacity6 = B0006.cycle(2).data.Capacity;
nom_capacity7 = B0007.cycle(2).data.Capacity;
nom_capacity18 = B0018.cycle(3).data.Capacity;

%% Extract the datapoints, moved charge, cycle indices. 

[~ , Y] = ExtractPartialCurve(B0005,3.6,0.2,4);
X = 1:length(Y);
Q = ExtractTotalMovedCharge(B0005);

X = transpose(X);
Y = transpose(Y);

%replace Capacity with SoH
Y = Y/nom_capacity5;

%%
clc
p_times = [166];

myfunction(p_times,X,Q,Y);

%% Train and Test, Model, Plots.

function myfunction(p_times, X, Q, Y)

    h1 = figure(1);
    plot(X,Y, '.', 'DisplayName',"Data");
    ylim([0.6 1])
    legend('location', 'best');

   
    h2 = figure(2);
    plot(X,Y, '.', 'DisplayName',"Data");
    ylim([0.6 1])
    legend('location', 'best');


    h3 = figure(3);  
    plot(X,Y, '.', 'DisplayName',"Data");
    ylim([0.6 1])
    legend('location', 'best');


    h4 = figure(4);
    plot(X,Y, '.', 'DisplayName',"Data");
    ylim([0.6 1])
    legend('location', 'best');
  

    % TRAIN-TEST DIVISION   
    pred_times = p_times; 
    colors = ["#fcba03" "#fc034e" "#782dfa"];
    
    for i = 1:length(pred_times)

        X_train = X(1:pred_times(i));
        Q_train = Q(1:pred_times(i));
        Y_train = Y(1:pred_times(i));
        
        X_test = X(pred_times(i)+1:end);
        Q_test = Q(pred_times(i)+1:end);
        Y_test = Y(pred_times(i)+1:end);
        
        % MODELS
        
        % linear model
        lin_f = fit(X_train,Y_train,'linear');
        
        % polynomial model
        poly_f = fit(X_train,Y_train,'poly3');
        
        % Hybrid model
        fitfun = fittype( @(a,b,c,d,x) a+b.*sqrt(x)+c.*(x)+d.*(x.^2));
        hybrid_f = fit(Q_train, Y_train, fitfun);
        
        % exp1 model
        exp1_f = fit(X_train,Y_train,'exp1')
        
        
        % exp2 model
        exp2_f = fit(X_train,Y_train,'exp2');
        
    
        % RESULTS
    
        y_lin = lin_f(X);
        y_poly = poly_f(X);
        y_hybr = hybrid_f(Q);
        y_exp1 = exp1_f(X);
        y_exp2 = exp2_f(X);
        
        R2_lin = Rsquared(Y, y_lin);
        R2_poly = Rsquared(Y, y_poly);
        R2_hybr = Rsquared(Y, y_hybr);
        R2_exp1 = Rsquared(Y, y_exp1);
        R2_exp2 = Rsquared(Y, y_exp2);
        
    
        % PLOTS
        
        %plot(X,y_lin,  'DisplayName',"linear R2: "+num2str(R2_lin));
        figure(1)
        hold on
        title ('Polynomial 3');
        xline(double(pred_times(i)), '-', 'color',colors(i), 'LineWidth',1.5, 'Label', "K = "+ num2str(pred_times(i)), 'DisplayName', "K = "+ num2str(pred_times(i)));
        %plot(X_test,y_poly(pred_times(i)+1:end),   'o','color',colors(i), 'DisplayName',"Poly3   R2: "+num2str(R2_poly));
        %plot(X_test,y_hybr(pred_times(i)+1:end), '.', 'color',colors(i),'DisplayName',"Hybrid   R2: "+num2str(R2_hybr));
        plot(X,y_poly,   '.','color',colors(i), 'DisplayName',"Poly3   R2: "+num2str(R2_poly));


        figure(2)
        hold on
        title ('Hybrid');
        xline(double(pred_times(i)), '-', 'color',colors(i), 'LineWidth',1.5, 'Label', "K = "+ num2str(pred_times(i)), 'DisplayName', "K = "+ num2str(pred_times(i)));
        plot(X,y_hybr, '.', 'color',colors(i),'DisplayName',"Hybrid   R2: "+num2str(R2_hybr));
     

        figure(3)
        hold on
        title ('Single Exponential');
        xline(double(pred_times(i)), '-', 'color',colors(i), 'LineWidth',1.5, 'Label', "K = "+ num2str(pred_times(i)), 'DisplayName', "K = "+ num2str(pred_times(i)));
        %plot(X_test,y_exp1(pred_times(i)+1:end), 'o','color',colors(i), 'DisplayName',"Exp1   R2: "+num2str(R2_exp1));
        %plot(X_test,y_exp2(pred_times(i)+1:end),'.',  'color',colors(i),'DisplayName',"Exp2   R2: "+num2str(R2_exp2));
        plot(X,y_exp1, '.','color',colors(i), 'DisplayName',"Exp1   R2: "+num2str(R2_exp1));


        figure(4)
        hold on
        title ('Double Exponential');
        xline(double(pred_times(i)), '-', 'color',colors(i), 'LineWidth',1.5, 'Label', "K = "+ num2str(pred_times(i)), 'DisplayName', "K = "+ num2str(pred_times(i)));
        plot(X,y_exp2,'.',  'color',colors(i),'DisplayName',"Exp2   R2: "+num2str(R2_exp2));
  
    end
end

 
%%

%plot(X_test, Y_test, 'ro' );

%{
model = fitrsvm(transpose(X_train), Y_train, 'Standardize', true, "KernelFunction","polynomial", PolynomialOrder=2); %, 'KernelFunction', 'polynomial');
pred = predict(model, transpose(X_test));
full_pred = predict(model, transpose(X));
err = loss(model,transpose(X_test), Y_test);
%}
%% Plots

%{
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

%}
%%
