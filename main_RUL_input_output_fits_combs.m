%% RUL prediction experiments
% Here we attempted RUL prediction by fitting the known part of the curve
% with a function, from which we generated 100 points to fit another
% function, from which we get the output SoH.
%Various input/output combinations tested
% prediction problem

clear all
clc
load('B0005.mat')


% Measure nominal capacities
nom_capacity5 = B0005.cycle(2).data.Capacity;


%% Extract the datapoints, moved charge, cycle indices. 

[~ , Y] = ExtractPartialCurve(B0005,3.6,0.2,4);
X = 1:length(Y);
Q = ExtractTotalMovedCharge(B0005);

X = transpose(X);
Y = transpose(Y);

%replace Capacity with SoH
Y = Y/nom_capacity5;
EoL_5= find(Y<0.75,1);
%1.4/nom_capacity5



%% Define inputs, outputs, hybrid funct. 
inputs = ["raw" "exp1" "exp2" "hybrid"];
outputs = ["hybrid" "poly2" "poly3" "poly1"];

%inputs = ["exp2"];

% Hybrid model
hybrid_fun = fittype( @(a,b,c,d,x) a+b.*sqrt(x)+c.*(x)+d.*(x.^2));


%ITerate over all combinations to make predictions
for in = inputs
    for out = outputs

        figure()
        plot(X, Y, ".b", 'MarkerSize', 15);
        hold on
        title ('In: '+ in + '   /   Out: '+out); 

        in_function = in;
        out_function = out;
        if out == "hybrid"
            out_function = hybrid_fun;
        end

        for i=80:20:120
            %Points division
            X_train = X(1:i);
            Q_train = Q(1:i);
            Y_train = Y(1:i);
            
            X_test = X(i:end);
            Q_test = Q(i:end);
            Y_test = Y(i:end);
        
            %Actual computation of fittings
            if in ~= "raw"
                in_fit = fit(X_train,Y_train, in_function);
                in_range = i/100:i/100:i;
                out_fit = fit(transpose(in_range), in_fit(in_range), out_function);
            else 
                out_fit = fit(X_train, Y_train, out_function);
            end
            
            
            %Doing plots
            plot(X, out_fit(X), '.-', 'MarkerSize',15);
            ylim([0.65 1])
            xline(i, '-', 'LineWidth',1.5, 'Label', "K = "+ num2str(i), 'DisplayName', "K = "+ num2str(i));
            xline(EoL_5, 'r--', 'LineWidth',1, 'Label', "EoL cycle = "+ num2str(EoL_5));
            yline(0.75, 'r--', 'LineWidth',1, 'Label', "EoL = "+ num2str(0.75));
            
        end
    end
end