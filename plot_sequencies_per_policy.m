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
%X2 = X2.X;
numObservation = numel(X);

if exist('d','var')==0
    d = MITLoadingCode();
    d(51).policy_readable = strcat(d(51).policy_readable, 'C');
end

%% Sort data by length of the sequences. This will allow to introduce less padding in the sequences belonging to the same batch
%This will introduce a peak in the training error, at the beginning of each
%epoch, due to the fact that all the longest squences comes there, and the
%shortest at the end

sequenceLengths = [];
for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,1);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
X = X(idx);
Y = Y(idx);

 figure()
 bar(sequenceLengths)
 xlabel("Battery sample",'FontSize',18 );
 ylabel("Length",'FontSize',18 );
 title("Battery life length in cycles")
 %matlab2tikz('..\..\thesis\Battery_length.tex');

 figure()
 histogram(sequenceLengths)
 xlabel("Life length", FontSize=18)
 ylabel("Battery samples", FontSize=18)
 title("Battery life length distribution")
 %matlab2tikz('..\..\thesis\Battery_length_distrib.tex');
  %%
for i = 85:124
    newPol = d(i).policy_readable;
    newPol = extractBetween(newPol, "","C-");
    newPol = strcat(newPol, "C");
    d(i).policy_readable = newPol;
end


%%
policies =cell(124,1);
capacities = cell(124,1);
tempcapacities = cell(124,1);
lengths = [];

for i=1:124
    policies{i} = d(i).policy_readable;
    capacities{i} = d(i).summary.QDischarge;
    lengths(i) = size(d(i).cycles,2);
end
policies = string(policies);
lengths = lengths';

G = findgroups(policies);
%%
[G,idx] = sort(G,'ascend');
lengths = lengths(idx);
policies = policies(idx);
for i=1:124
    tempcapacities{i} = capacities{idx(i)};
end
capacities = tempcapacities;
%%
counts = hist(G,1:numel(G));
result = [1:numel(G); counts];

[result(2,:),idx] = sort(result(2,:),'descend');
result(1,:) = result(1,idx);
result(:, 11:end) = [];
%% Subplots

figure()
diff = 4;
for i=5:8 %size(result,2)
    subplots(i-diff) =  subplot(2,2,i-diff);
    hold on
    %if i-diff == 2
        xlabel("Cycles")
    %end
    if mod(i,2) == 1
        ylabel("Capacity (Ah)")
    end

    idx = find(G == result(1,i));
    title("Policy " + policies(idx(1)))
    for j=1:numel(idx)
        plot(capacities{idx(j)});
    end
    cleanfigure ('minimumPointsDistance', 10);
    hold off

end

pos = get(subplots(1),'Position');
set(subplots(1),'Position', pos + [0 0 0.09 -0.1]);

pos = get(subplots(2),'Position');
set(subplots(2),'Position', pos + [0.03 0 0.09 -0.1]);

pos = get(subplots(3),'Position');
set(subplots(3),'Position', pos + [0 0 0.09 -0.1]);

pos = get(subplots(4),'Position');
set(subplots(4),'Position', pos + [0.03 0 0.09 -0.1])

% pos = get(subplots(3),'Position');
% set(subplots(3),'Position', pos + [0 -.24 0 0]);
% 
% pos = get(subplots(4),'Position');
% set(subplots(4),'Position', pos + [0 -.36 0 0]);


%matlab2tikz('..\..\thesis\sequencies_by_policy_1.tex');

%% Normal plots

diff = 0;
for i=2:2 %size(result,2)
    figure()
    hold on
  
    xlabel("Cycles")
    ylabel("Capacity (Ah)")

    idx = find(G == result(1,i));
    title("Policy " + policies(idx(1)))
    for j=1:numel(idx)
        plot(capacities{idx(j)});
    end
   % cleanfigure ('minimumPointsDistance', 10);
    pbaspect([8 2 2])
    hold off
    cleanfigure ('minimumPointsDistance', 10);
    matlab2tikz('..\..\thesis\sequencies_by_policy_2_2.tex');
end
