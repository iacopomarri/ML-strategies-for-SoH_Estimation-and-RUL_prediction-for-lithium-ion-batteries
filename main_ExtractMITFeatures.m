%% Use MIT code to load and clean and fix dirty data. Extract 3 features from MIT paper
% mit code
clearvars  -except d; 
clc


if exist('d','var')==0
    d = MITLoadingCode();
end

numObservation = numel(d);

%%
plot(d(1).cycles(1).V)
xlabel('Time') 
ylabel('Voltage (V)');
%cleanfigure('minimumPointsDistance', 0.1)
%matlab2tikz('..\..\thesis\MIT_voltage_profile.tex');

plot(d(20).summary.QDischarge)
xlabel('Timesteps (cycles)') 
ylabel('Capacity (Ah)');
%cleanfigure('minimumPointsDistance', 0.1)
%matlab2tikz('..\..\thesis\MIT_degradation_sample.tex');


%% Extract discharge curve for all cycles, subtract with cycle 10 --> deltas. 
clc
close all

numObservation = numel(d);
%numObservation = 5;
numFeatures = 3;

%This contains all the difference between curves at cycle 10 and each other
%cycle k, for each batt
deltas = cell(numObservation,1);


figure()
hold on
ylabel('Voltage (V)','FontSize',18 );
xlabel('DeltaQ','FontSize',18 );
%f1 = figure();
%f2 = figure();
idx = [];
voltage_interp = linspace(3.25, 3.4, 2000);
%voltage_interp = linspace(2.8, 3.1, 2000);
%voltage_interp = linspace(2.8, 3, 2000);

for batt = 1:numObservation 
    sample = d(batt);
    
    %Extract cycle 10, clean initial and last part (before max, after min)
    Qdis_i = sample.cycles(10).Qd;
    Vdis_i = sample.cycles(10).V;
   
    [maxim, index] = max(Vdis_i);
    Qdis_i(1:index-1) = [];
    Vdis_i(1:index-1) = [];
    
    [minim, index] = min(Vdis_i);
    Qdis_i(index+1:end) = [];
    Vdis_i(index+1:end) = [];

    %Make voltage array values unique, needed for interpolation
    [Vdis_i,ia] = unique(Vdis_i, "stable"); 
    Qdis_i = Qdis_i(ia);

    %Interpolate voltage of initial cycle
    interpolation = fit(Vdis_i, Qdis_i, 'linear');
    Qinterp_1 = interpolation(voltage_interp);

%        figure()
%        hold on
%        ylabel('Discharge capacity Q (Ah)','FontSize',18  );
%         xlabel('Voltage (V)','FontSize',18 );
%          legend('location', 'southwest','FontSize',12);
%           plot(voltage_interp, Qinterp_1, 'DisplayName','Cycle 10');

    for k =1:size(sample.cycles, 2)
        %Extract cycle k, clean initial and last part
        Qdis_k = sample.cycles(k).Qd;
        Vdis_k = sample.cycles(k).V;
             
        [maxim, index] = max(Vdis_k);
        Qdis_k(1:index-1) = [];
        Vdis_k(1:index-1) = [];
        
        [minim, index] = min(Vdis_k);
        Qdis_k(index+1:end) = [];
        Vdis_k(index+1:end) = [];
    
    
        %Make voltage array values unique, needed for interpolation
        [Vdis_k,ia] = unique(Vdis_k, "stable"); 
        Qdis_k = Qdis_k(ia);
        
        %Interpolate voltage of prediction cycle
        interpolation = fit(Vdis_k, Qdis_k, 'linear');
        Qinterp_k = interpolation(voltage_interp);
    
%         if k==100
%             plot(voltage_interp,Qinterp_k, 'DisplayName',"Cycle 100");          
%         end
% 
%         if k==300
%             plot(voltage_interp,Qinterp_k, 'DisplayName',"Cycle 300");          
%         end
% 
%         if k==500
%             plot(voltage_interp,Qinterp_k, 'DisplayName',"Cycle 500");          
%         end
% 
%         if k==700
%             plot(voltage_interp,Qinterp_k, 'DisplayName',"Cycle 700");          
%         end
       
        %compute diff curve.
        deltaQ = Qinterp_k - Qinterp_1; 
        
        deltas{batt}(k,:) = deltaQ;
        
        %set k to the n cycle you want to plot
%         if(k==100)
%             figure()
%            plot(deltaQ, voltage_interp);
%         end
    end
end
%%
figure()
hold on
for b=1:numObservation
    plot(deltas{b}(100,:), voltage_interp)
end

savefig("../../RUL features tries/2.4_2.6 V/delta_curves_2.4_2.6.fig");
%% PLOT stuff
% clearvars -except deltas d
% load("deltas.mat");
% numObservation = size(deltas,1);
% voltage_interp = linspace(2.05, 3.55, 2000);
% 
% figure()
% hold on
% 
% 
% k=100; % Keep k > 100
% for i=1:numObservation
%     if size(deltas{i},1)> k
%         plot(deltas{i}(k,:), voltage_interp);
%     end
% end

% cleanfigure('minimumPointsDistance', 0.1)
% matlab2tikz('..\..\thesis\deltas.tex');



%% compute the 3 final features. Each formula can be found in supplementary info pdf of MIT paper
clc

X = cell(numObservation, 1);
Y = cell(numObservation,1);

for i = 1:numObservation 
    sample = d(i);

    var = [];
    mindelta = [];
    tempint = [];

    for k = 10:size(sample.cycles, 2)
      delta = deltas{i}(k,:);
            
      %1st  Min delta 
      mindelta(k) = log10(abs(min(delta))); 
      
      %2nd Integral Temperature 
      tempint(k) =  sum(sample.summary.Tavg(2:k)); 
    
      %3rd Log variance 
      avg = mean(delta);
      delta = delta - avg;
      delta = delta.^2;
      delta = sum(delta)/numel(delta);
      delta = abs(delta);
      delta = log10(delta);
         
      var(k) = delta; 
    end
    
    %put 0 instead of -inf that is computed by log10(0)
    var(10) = 0;
    mindelta(10) = 0;

    X{i} =[var' mindelta' tempint'];
    X{i} = transpose(X{i});

    Y{i} = flip(sample.summary.cycle);
    %%Y{i} = flip()
    Y{i} = transpose(Y{i});
 
end
%% Set first 10 values of each ftrs equal to 11th.

for i=1:numObservation
   

        X{i}(1,1:10) = X{i}(1,11);
        X{i}(2,1:10) = X{i}(2,11);
        X{i}(3,1:10) = X{i}(3,11);

end


%% Sort data by length of the sequences. This will allow to introduce less padding in the sequences belonging to the same batch
%This will introduce a peak in the training error, at the beginning of each
%epoch, due to the fact that all the longest squences comes there, and the
%shortest at the end

for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
X = X(idx);
Y = Y(idx);



%% Plot the 100th value of the log var feature, for each batt. to compare it with MIT paper

variance = [];
mindelta = [];
tempint = [];
sequenceLengths = [];

%cycle to plot
k=700

% X = Xtemp
% %REMOVE THIS AFTER
% for i=1:124
%     X{i} = X{i}';
% end

for i=1:size(X,1)
    sequence = X{i};
    %sequenceLengths(i) = size(sequence,2);

    if size(X{i}, 2) >  k
        sequenceLengths = [sequenceLengths size(sequence,2)];
        variance = [variance X{i}(1,k)];
        mindelta = [mindelta X{i}(2,k)];
        tempint = [tempint X{i}(3,k)];
    end
end
a = linspace(5,10,length(variance));

figure()
hold on
ylabel('Log10  Sequence Length','FontSize',18 );
xlabel('Log10  Var(DeltaQ)','FontSize',18 );
scatter(variance, log10(sequenceLengths),50, a,"filled");
savefig("../../RUL features tries/3.25_3.4 V/var_3.25_3.4.fig");

figure()
hold on
ylabel('Log10  Sequence Length','FontSize',18 );
xlabel('Log10  Min(DeltaQ)','FontSize',18 );
scatter(mindelta, log10(sequenceLengths),50, a,"filled");

savefig("../../RUL features tries/3.25_3.4 V/min_3.25_3.4.fig");

figure()
hold on


scatter(tempint, log10(sequenceLengths),50, a, "filled");
ylabel('Log10  Sequence Length','FontSize',18 );
xlabel('Temperature sum (C°)','FontSize',18 );
%savefig("../../RUL features tries/3.25_3.4 V/temp_3.25_3.4.fig");

%%
%save("Partial_MIT_features_3 to 3,4.mat", "X", "Y");
%%

for i=1:124
    X{i} = X{i}';
end
save("../../RUL features tries/3.25_3.4 V/Partial_MIT_features_3,25 to 3,4.mat", "X", "Y");
