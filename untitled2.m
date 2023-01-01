%%  Load MIT feature (variance of DeltaQ curves, min of DeltaQ curves, integral temperature)    
clc 
%clear all
clearvars -except d net
close all

rng("default")
load("MIT_Disch_Features.mat")


ftr_idx = [1 2 3 4]; %variance deltaQ, integral temp, num cycle
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

% mu = mean([X{:}],2);
% 
% sig = std([X{:}],0,2);
% 
% for i = 1:numel(X)
%     X{i} = (X{i} - mu) ./ sig;
% end


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
%Y = Y(idx);
    
figure
bar(sequenceLengths, "FaceColor","#e9a747")
xlabel("Sample")
ylabel("Length (cycles)")
%title("Sorted Data")




%% Plot the 700th value of the log var feature, for each batt. to compare it with MIT paper

variance = [];
mindelta = [];
tempint = [];
sequenceLengths = [];

%cycle to plot
k=700

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

scatter(variance, log10(sequenceLengths),50, a,"filled");
ylabel('Log10  Sequence Length (cycles)','FontSize',18 );
%xlabel('Log10  Var(DeltaQ)','FontSize',18 );

xlabel(num2str("Log10  Var(DeltaQ), " + k +"th cycle" ),'FontSize',18 );

figure()
hold on

scatter(mindelta, log10(sequenceLengths),50, a,"filled");
ylabel('Log10  Sequence Length (cycles)','FontSize',18 );
%xlabel('Log10  Min(DeltaQ)','FontSize',18 );

xlabel(num2str("Log10  Min(DeltaQ), " + k +"th cycle" ),'FontSize',18 );


figure()
hold on

scatter(tempint, log10(sequenceLengths),50, a, "filled");
ylabel('Log10  Sequence Length (cycles)','FontSize',18 );
%xlabel('Temperature sum (C°), cycle','FontSize',18 );

xlabel(num2str("Temperature sum (C°), " + k +"th cycle" ),'FontSize',18 );

%% Backup data to normalize them



normalizedX = X;

%% Transpose dataset to then normalize it
for i=1:numObservation
    normalizedX{i} = normalizedX{i}';
end


mu = mean([normalizedX{:}],2);

sig = std([normalizedX{:}],0,2);

for i = 1:numel(normalizedX)
    normalizedX{i} = (normalizedX{i} - mu) ./ sig;
end


% Transpose dataset back to normal
for i=1:numObservation
    X{i} = X{i}';
end





%% Plot the 700th value of the log var feature, for each batt. to compare it with MIT paper

variance = [];
mindelta = [];
tempint = [];
sequenceLengths = [];

%cycle to plot
k=700

for i=1:size(normalizedX,1)
    sequence = normalizedX{i};
    %sequenceLengths(i) = size(sequence,2);

    if size(normalizedX{i}, 2) >  k
        sequenceLengths = [sequenceLengths size(sequence,2)];
        variance = [variance normalizedX{i}(1,k)];
        mindelta = [mindelta normalizedX{i}(2,k)];
        tempint = [tempint normalizedX{i}(3,k)];
    end
end
a = linspace(5,10,length(variance));

figure()
hold on

scatter(variance, log10(sequenceLengths),50, a,"filled");
ylabel('Log10  Sequence Length (cycles)','FontSize',18 );
%xlabel('Log10  Var(DeltaQ)','FontSize',18 );

xlabel(num2str("Log10  Var(DeltaQ), " + k +"th cycle" ),'FontSize',18 );

figure()
hold on

scatter(mindelta, log10(sequenceLengths),50, a,"filled");
ylabel('Log10  Sequence Length (cycles)','FontSize',18 );
%xlabel('Log10  Min(DeltaQ)','FontSize',18 );

xlabel(num2str("Log10  Min(DeltaQ), " + k +"th cycle" ),'FontSize',18 );


figure()
hold on

scatter(tempint, log10(sequenceLengths),50, a, "filled");
ylabel('Log10  Sequence Length (cycles)','FontSize',18 );
%xlabel('Temperature sum (C°), cycle','FontSize',18 );

xlabel(num2str("Temperature sum (C°), " + k +"th cycle" ),'FontSize',18 );
