%% Cross Validation for different voltage windows, to find for each, best model, best HP.
clc
close all
clearvars  -except d; 
rng(3) 

%% Load data

removeIrregularities = true;
load("MIT_PCC_Features.mat"); %Load this only for Ysoh
numWindow = [3 3.4];  %change this to perform the process on a different window

stringWindow = num2str(numWindow(1) + "_" +num2str(numWindow(2)) + " V");
dashedWindow = num2str(numWindow(1) + "-" +num2str(numWindow(2)) + " V");
%X = load("MIT_features.mat").X;

X = load("../../Papers/RUL features tries/"+ stringWindow +"/Partial_MIT_features_3 to 3,4.mat").X;


Y = Y_SoH;
numObservation = numel(X);
clear Y_RUL


%% Find noisy samples, plot and remove them   4  71  110   117

if(removeIrregularities)
    %plot all SoHs
    %figure()
    %hold on
    for i=1:numel(Y)%for i=idx
        %plot(Y{i})
        %plot(X{i}(:,[1]))
    end
    
    %find irregularities (if present)
    idx = [];
    for i=1:numel(Y)
        if find(Y{i}>1.1)
            idx = [idx i];
        end
    end
    
    %plot irregularities
    %figure()
    %hold on
    for i=idx
       % plot(Y{i})
        %plot(X{i}(:,[1]))
    end
    
    %remove irregularities
    Y(idx) = [];
    X(idx) = [];
    Y_SoH(idx) = [];
    
    %recompute num obsv
    numObservation = numel(X);
end

%%  4 60 71 72  89 110 117          noisy: 4  71  110   117   weird shape: 60 72 89
%close all

%order sequencies
sequenceLengths = [];
for i=1:numel(X)
    sequence = X{i};
    sequenceLengths(i) = size(sequence,1);
end

[sequenceLengths,idx] = sort(sequenceLengths,'descend');
X = X(idx);
Y_SoH= Y_SoH(idx);
policy = policy(idx);


% figure ()
% hold on
% for i=[60,  72, 89]
%     plot(Y_SoH{i})
% endi
clear = [4, 60, 72, 89, 71, 110, 117];
noise = [4, 71, 110, 117];
weird = [60, 72, 89];

indices = (1:numObservation);
clean = indices;
no_weird = indices;
no_noise = indices;

for i=1:size(clear,2)
    clean = clean(find(clean~=clear(i)));
end

for i=1:size(weird,2)
    no_weird = no_weird(find(no_weird~=weird(i)));
end

for i=1:size(noise,2)
    no_noise = no_noise(find(no_noise~=noise(i)));
end

%% INTERPOLATE THE CURVES, SAVE EM IN "normalizedX"
clc
normalizedX = {}

figure()
hold on
fitting_freq = 2000;
for i=clean
    interp_x = 1:size(Y_SoH{i},2);
    len_interp = linspace(0, interp_x(end), fitting_freq);
    
    
    interpolation = fit(interp_x', smooth(Y_SoH{i}'),'linear');
    interp_y = interpolation(len_interp);

    plot(interp_y, color="b")
    normalizedX{i} = interp_y'
end

for i=weird
    interp_x = 1:size(Y_SoH{i},2);
    len_interp = linspace(0, interp_x(end), fitting_freq);
    
    
    interpolation = fit(interp_x', smooth(Y_SoH{i}'),'linear');
    interp_y = interpolation(len_interp);

    plot(interp_y, color="r")
    normalizedX{i} = interp_y'
end

%%  DEFINE GROUPS 1 AND 2 FOR NORMAL CURVES AND FOR THE 3 ABNORMAL CURVES
prove = []
groups = []
j=1
for i=clean
    prove =vertcat(prove, normalizedX{i})
    groups(j) = 1
    j = j+1
end
       
for i=weird
    prove = vertcat(prove, normalizedX{i})
    groups(j) = 2
    j = j+1
end

figure
hold on
plot(prove(118,:))
plot(prove(119,:))
plot(prove(120,:))


%% PARALLELCOORDS WITH 1 GROUP, AND 3 SINGLE PLOTS FOR THE WEIRD CURVES

% figure()
% hold on
% parallelcoords(prove,'quantile',.1,'LineWidth',2)
% title("quantile 0.1")
% for i=weird
%     plot(normalizedX{i}, color="r")
% end
% 
% 
% figure()
% hold on
% parallelcoords(prove,'quantile',.2,'LineWidth',2)
% title("quantile 0.2")
% for i=weird
%     plot(normalizedX{i}, color="r")
% end
% 
% figure()
% hold on
% parallelcoords(prove,'quantile',.25,'LineWidth',2)
% title("quantile 0.25")
% for i=weird
%     plot(normalizedX{i}, color="r")
% end

% figure
% hold on
% plot(Y_SoH{1})

%% %% PARALLELCOORDS WITH 2 GROUPS, 1 FOR NORMAL, 2 FOR WEIRD

% figure()
% hold on
% parallelcoords(prove,'LineWidth',2, group=groups)
% title("quantile 0.1")
% % for i=weird
% %     plot(normalizedX{i}, color="r")
% % end


figure()
hold on
parallelcoords(prove,'quantile',.2,'LineWidth',2, group=groups)
title("quantile 0.2")
% for i=weird
%     plot(normalizedX{i}, color="r")
% end

% figure()
% hold on
% parallelcoords(prove,'quantile',.25,'LineWidth',2, group=groups)
% title("quantile 0.25")
% % for i=weird
% %     plot(normalizedX{i}, color="r")
% % end

% figure
% hold on
% plot(Y_SoH{1})