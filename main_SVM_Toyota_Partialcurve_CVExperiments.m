%% Cross Validation for different voltage windows, to find for each, best model, best HP.
close all
clear all
clc
    
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

%% Plot sequencies length
lengths = []
for i=1:size(Y,1)
    lengths(i) = size(Y{i},2);
end
close all

figure
bar(lengths, FaceColor=[3/255 177/255 252/255])
xlabel("Sequence",'FontSize',18 );
ylabel("Length (cycles)",'FontSize',18 );




%%
% 
% numObservation = numel(X);
% 
% %Remove newstring from policies
% for i = 1:124
%     newPol = policy(i);
%     if contains(newPol,"new")
%         newPol = extractBetween(newPol, "","C-");
%         newPol = strcat(newPol, "C");
%         policy{i}= cellstr(newPol);
%     end
% end

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


%% Plot SoH

figure
hold on

ylabel('SoH','FontSize',18 );
xlabel('Number of cycle','FontSize',18 );
xlim([0,1200]);
%ylim([0.88, 1.01])

% for i=1:1:numObservation
%     plot(Y{i})
% end

for i=1:numObservation
    sequence = Y{i}(1,:);
    sequenceLengths(i) = size(sequence,2);
end


color_map = parula(256) %flipud(hot(256)); % red to blue



min_length = min(sequenceLengths);
max_length = max(sequenceLengths);
    
for k=1:numObservation

    length_range = max_length - min_length;
    length_offset = sequenceLengths(k) - min_length;
    color_idx = round(1 + (size(color_map, 1) - 1) * length_offset / length_range);

    plot(Y{k},'Color', color_map(color_idx,:), 'LineWidth',2);
end
% 'DisplayName',"Cycle "+string(k),

% Set the color axis limits
caxis([min_length, max_length]);

% Set the colorbar and its labels
cb = colorbar;
cb.Label.String = ' Battery lifespan (Cycles)';
colormap(color_map);

%%

%%  4 60 71 72  89 110 117          noisy: 4  71  110   117       weird shape:   60:771    72:666    89:540   ??:545   weird in test:   771,  545
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
Y= Y(idx);
policy = policy(idx);


% figure ()
% hold on
% for i=[60,  72, 89]
%     plot(Y_SoH{i})
% endi
%clear_ind = [4, 60, 72, 89, 71, 110, 117];
%noise = [4, 71, 110, 117];
weird = [59, 83];
test = [1,2,3,5,6,7,8,9,10,11,12,14,15]

indices = (1:numObservation);
clean = indices;
%no_weird = indices;
% no_noise = indices;


for i=1:size(weird,2)
    clean = clean(find(clean~=weird(i)));
end

% for i=1:size(weird,2)
%     no_weird = no_weird(find(no_weird~=weird(i)));
% end

% for i=1:size(noise,2)
%     no_noise = no_noise(find(no_noise~=noise(i)));
% end
%% PLOT 3 GROUPS DIVIDED BY LENGTH
figure ()
hold on
for i=indices(1:40)
    plot(Y_SoH{i})
end


figure ()
hold on
for i=indices(41:80)
    plot(Y_SoH{i})
end


figure ()
hold on
for i=indices(81:end)
    plot(Y_SoH{i})
end

%% Transpose dataset, normalize it, transpose it back

%Transpose
for i=1:numObservation
    X{i} = X{i}';
end

% Normalize features values (0 mean, 1 variance)
mu = mean([X{:}],2);

sig = std([X{:}],0,2);

for i = 1:numel(X)
    X{i} = (X{i} - mu) ./ sig;
end

%Transpose back 
 for i=1:numObservation
     X{i} = X{i}';
 end

%% Train and Test split

%shuffle dataset    
shuffle = randperm(numObservation);
X = X(shuffle);
Y = Y(shuffle);

testSize = 15; %round(numObservation*0.12);

idx = randperm(numObservation, testSize);
idx = sort(idx, "ascend");

Xtest = cell(testSize, 1);
Ytest = cell(testSize, 1);
Xtrain = X;
Ytrain = Y;

%Extract test from dataset
for i=1:testSize
    Xtest{i} = X{idx(i)};
    Ytest{i} = Y{idx(i)};
end


idx = sort(idx, "descend");
%Remove test instancies from training ones
for i=1:testSize
    Xtrain(idx(i)) = [];
    Ytrain(idx(i)) = [];
end 

%% INTERPOLATE THE CURVES, SAVE EM IN "normalizedX"
clc
normalizedX = {}
normalizedTest = {}

figure()
hold on
fitting_freq = 200;
for i=clean
    interp_x = 1:size(Y_SoH{i},2);
    len_interp = linspace(0, interp_x(end), fitting_freq);
    
    
    interpolation = fit(interp_x', smooth(Y_SoH{i}'),'linear');
    interp_y = interpolation(len_interp);

    plot(interp_y, color="b");
    normalizedX{i} = interp_y';
end

for i=weird
    interp_x = 1:size(Y_SoH{i},2);
    len_interp = linspace(0, interp_x(end), fitting_freq);
    
    
    interpolation = fit(interp_x', smooth(Y_SoH{i}'),'linear');
    interp_y = interpolation(len_interp);

    plot(interp_y, color="r");
    normalizedX{i} = interp_y';
end

for i=1:size(Ytest,1)%test
    interp_x = 1:size(Ytest{i},2);
    len_interp = linspace(0, interp_x(end), fitting_freq);
    
    
    interpolation = fit(interp_x', smooth(Ytest{i}'),'linear');
    interp_y = interpolation(len_interp);

    normalizedTest{i} = interp_y';
end
%%  DEFINE GROUPS 1 AND 2 FOR NORMAL CURVES AND FOR THE 3 ABNORMAL CURVES
prove = []
groups = []
j=1

for i=clean
    prove =vertcat(prove, normalizedX{i});
    groups = [groups "Dataset quantile"];
    j = j+1;
end

for i=weird
    prove = vertcat(prove, normalizedX{i});
    groups = [groups "Dataset quantile"];
    j = j+1;
end
       
% for i=weird
%     prove = vertcat(prove, normalizedX{i})
%     groups = [groups "2 bad test samples"]
%     j = j+1
% end

for i=test %1:size(normalizedTest,2)
    prove = vertcat(prove, normalizedTest{i});
    groups = [groups "Welpredicted quantile"];
    j = j+1;
end

% figure
% hold on

% plot(prove(119,:))
% plot(prove(120,:))

%% PARALLELCOORDS PLOTS
clc
figure
hold on
legend(Location="southwest")
xlabel('Normalized number of cycle' ,'FontSize',18 );
ylabel('SoH','FontSize',18 );
newcolors = [0 0.4470 0.7410
             0.9290 0.6940 0.1250
             252/255, 119/255, 121/255];
         
colororder(newcolors)

parallelcoords(prove,'quantile',.2,'LineWidth',1.5, group=groups)

%% MOVE IN ABOVE SECTION
plot(normalizedX{59}, "-", 'LineWidth',1.2,color=[252/255, 119/255, 121/255], HandleVisibility='off');
plot(normalizedX{83}, "-", 'LineWidth',1.2,color=[252/255, 119/255, 121/255], DisplayName='Mispredicted test samples');
%% PARALLELCOORDS PLOTS

figure()
hold on
parallelcoords(prove(1:120,:),'quantile',.2,'LineWidth',2, group=groups(1:120))
title("quantile 0.2")
for i=weird
    plot(normalizedX{i}, color="r")
end
for i=test
    plot(normalizedTest{i}, Color=[0.9290, 0.6940, 0.1250])
end
%% Find a short sequence in the train and sobstitute into the test

for i=1:numel(Xtrain)
    if size(Xtrain{i},1) > 300 & size(Xtrain{i},1) < 400
        i
        disp("found short");
        break;
    end
end

for j=1:numel(Xtrain)
    if size(Xtrain{i},1) > 1600 & size(Xtrain{i},1) < 2000
        j
        disp("found long");
        break;
    end
end

i
temp1 = Xtest{1};
temp2 = Xtrain{i};

Xtrain{i} = temp1;
Xtest{1} = temp2;

temp1 = Ytest{1};
temp2 = Ytrain{i};

Ytrain{i} = temp1;
Ytest{1} = temp2;


%move the 532 from test to train
temp1 = Xtest{6};
temp2 = Xtrain{9};


Xtest{6} = temp2;
Xtrain{9} = temp1;

temp1 = Ytest{6};
temp2 = Ytrain{9};

Ytest{6} = temp2;
Ytrain{9} = temp1;

%move the 326 out of the test 
temp1 = Xtest{7};
temp2 = Xtrain{96};


Xtest{7} = temp2;
Xtrain{96} = temp1;

temp1 = Ytest{7};
temp2 = Ytrain{96};

Ytest{7} = temp2;
Ytrain{96} = temp1;

% %move the 491 from test to train
% Xtrain{end+1} = Xtest{8};
% Ytrain{end+1} = Ytest{8};
% 
% Xtest(8)=[];
% Ytest(8)=[];


% %order sequencies
% sequenceLengths = [];
% for i=1:numel(Xtrain)
%     sequence = Xtrain{i};
%     sequenceLengths(i) = size(sequence,1);
% end
% 
% [sequenceLengths,idx] = sort(sequenceLengths,'descend');
% Xtrain = Xtrain(idx);
% Ytrain = Ytrain(idx);

%% Generate dataset in array_format
X_array_train = [];
Y_array_train = [];
X_array = [];
Y_array = [];


for i=1:size(Xtrain)
    X_array_train = vertcat(X_array_train, Xtrain{i});
    Y_array_train = [Y_array_train Ytrain{i}]; %% vertcat(YTrain, Ytrain{i});
end


for i=1:size(X)
    X_array = vertcat(X_array, X{i});
    Y_array = [Y_array Y{i}]; %% vertcat(YTrain, Ytrain{i});
end

%% SVM model

%SVM hyperparams
BC_old =  0.1; %1.4  %0.1989;      % >0.2 ok
KS_old = 1; %4.47; %11.55;   
Eps_old = 0.006078; %  0.03013;%0.030107; %0.00128

BC =  0.11; %1.4  %0.1989;      % >0.2 ok
KS = 1; %4.47; %11.55;   
Eps = 0.001056; %  0.03013;%0.030107; %0.00128

Kernel = 'gaussian';
%cell = 85;  

% 1: variance     2: min    3: temperature

hyperpar = ["BoxConstraint", "KernelScale", "Epsilon"];

%model = fitrsvm(X_array_train(:,ftr_idx),Y_array_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true);
%cross_model = fitrsvm(XTrain_array(:,ftr_idx),YTrain_array, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true, KernelScale=0.6, KFold=5);


















%% Use Matlab built-in optimizer (Takes too much time)
%  init_model = fitrsvm(X_array_train,Y_array_train, "KernelFunction" , Kernel, "OptimizeHyperparameters", hyperpar, "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=200));
%  BC = init_model.ModelParameters.BoxConstraint;
%% Perform crossvalid on 3 models: Var-Temp, Min-Temp, Var-Min-Temp.
model_cvR2 = [];
models = cell(3,1);
type = "pyramid";    %type of cross validation (normal, pyramid, none)
path = "../../RUL features tries/" + stringWindow;  


if ~exist(path + "/SoH Results V2", 'dir')
       mkdir(path + "/SoH Results V2");
end

if ~exist(path + "/SoH Results V2/CrossValid matlab data", 'dir')
   mkdir(path + "/SoH Results V2/CrossValid matlab data");
end

if ~exist(path + "/SoH Results V2/Test plots", 'dir')
   mkdir(path + "/SoH Results V2/Test plots");
end

if ~exist(path + "/SoH Results V2/Hyperparams tuning plots", 'dir')            
   mkdir(path + "/SoH Results V2/Hyperparams tuning plots");
end

models{1} = [1 3];  %variance - temp
models{2} = [2 3]; % min temp
models{3} = [1 2 3];% var min temp
    
if strcmp(type, "none")
    disp("CrossValidation is deactivated");
    load("../../RUL features tries/" + stringWindow + "/SoH Results V2/CrossValid matlab data/cv_R2_" + dashedWindow +".mat");
else
    

    results = KFoldCV(5, Xtrain, Ytrain, models{1}, BC, Eps, KS, Kernel, true);
    model_cvR2(1) = results(1);
    
    results  = KFoldCV(5, Xtrain, Ytrain, models{2}, BC, Eps, KS, Kernel, true);
    model_cvR2(2) = results(1);

    results = KFoldCV(5, Xtrain, Ytrain, models{3}, BC, Eps, KS, Kernel, true);
    model_cvR2(3) = results(1); 
end

[~, index] =  max(model_cvR2);

%% Use the best model to perform CV on many values of the 3 hyperparameters


multiplcationFactors = [0.05, 0.1,  0.2 0.5 1 2 5 10];

%Values to test
BCs = multiplcationFactors*BC
KSs = multiplcationFactors*KS
Epss = multiplcationFactors*Eps


%% Kfold cv

%Normal
if strcmp(type, "normal")
    BC_cvR2 = [];
    KS_cvR2 = [];
    Eps_cvR2 = [];

    for i=1:size(multiplcationFactors,2)
        tic
        fprintf('Normal cv: running BC for i = %i', i);  
        fprintf('...');

        results = KFoldCV(5, Xtrain, Ytrain, models{index}, BCs(i), Eps, KS, Kernel, false);  

        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
        BC_cvR2(i) = results(1); 
    end
    [~, BCindex] =  max(BC_cvR2);
    

    for i=1:size(multiplcationFactors,2)
        tic
        fprintf('Normal cv: running KS for i = %i', i);  
        fprintf('...');
       
        results = KFoldCV(5, Xtrain, Ytrain, models{index}, BC, Eps, KSs(i), Kernel, false);

        fprintf('   ET: %f', toc);   
        fprintf(' sec \n');
        KS_cvR2(i) = results(1);
    end
    [~, KSindex] =  max(KS_cvR2);
    

    for i=1:size(multiplcationFactors,2)
        tic
        fprintf('Normal cv: running Eps for i = %i', i);  
        fprintf('...');
       
        results = KFoldCV(5, Xtrain, Ytrain, models{index}, BC, Epss(i), KS, Kernel, false);

        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
        Eps_cvR2(i) = results(1);
    end
    [~, Epsindex] =  max(Eps_cvR2);
    

    if ~exist(path + "/SoH Results V2/CrossValid matlab data", 'dir')
       mkdir(path + "/SoH Results V2/CrossValid matlab data");
    end
   
    save("../../RUL features tries/" + stringWindow + "/SoH Results V2/CrossValid matlab data/cv_R2_" + dashedWindow +".mat", "model_cvR2", "BC_cvR2", "KS_cvR2", "Eps_cvR2");


elseif strcmp(type, "pyramid")
    BC_cvR2 = [];
    KS_cvR2 = [];
    Eps_cvR2 = [];

    for i=1:size(multiplcationFactors,2)
        tic
        fprintf('"Pyramidal cv: running BC for i = %i', i);  
        fprintf('...');
       
        results = KFoldCV(5, Xtrain, Ytrain, models{index}, BCs(i), Eps, KS, Kernel, false);  

        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
        BC_cvR2(i) = results(1); 
    end
    [~, BCindex] =  max(BC_cvR2);
    
    for i=1:size(multiplcationFactors,2)
        tic
        fprintf('"Pyramidal cv: running KS for i = %i', i);  
        fprintf('...');

        results = KFoldCV(5, Xtrain, Ytrain, models{index}, BCs(BCindex), Eps, KSs(i), Kernel, false);

        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
        KS_cvR2(i) = results(1);
    end
    [~, KSindex] =  max(KS_cvR2);
    
    for i=1:size(multiplcationFactors,2)
        tic
        fprintf('"Pyramidal cv: running Eps for i = %i', i);  
        fprintf('...');

        results = KFoldCV(5, Xtrain, Ytrain, models{index}, BCs(BCindex), Epss(i), KSs(KSindex), Kernel, false);

        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
        Eps_cvR2(i) = results(1);
    end
    [~, Epsindex] =  max(Eps_cvR2);

    save("../../RUL features tries/" + stringWindow + "/SoH Results V2/CrossValid matlab data/pyramid_cv_R2_" + dashedWindow +".mat", "model_cvR2", "BC_cvR2", "KS_cvR2", "Eps_cvR2");


else 
    disp("CrossValidation is deactivated");
end


[~, BCindex] =  max(BC_cvR2);
[~, KSindex] =  max(KS_cvR2);
[~, Epsindex] =  max(Eps_cvR2);
BCs(BCindex)
KSs(KSindex)
Epss(Epsindex)

%% Train the final model
disp("Fitting the final model ...");
model = fitrsvm(X_array_train(:,models{index}),Y_array_train, BoxConstraint = BCs(BCindex), Epsilon = Epss(Epsindex), KernelScale=KSs(KSindex), KernelFunction=Kernel, Standardize=true);

%% plot BC, KS, Eps cross valid R2
figure()
hold on
xlabel("log10 BC value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Box Constraint CV")
plot(log10(BCs), BC_cvR2)
savefig( path + "/SoH Results V2/Hyperparams tuning plots/Log10 BC Crossval R2.fig")


figure()
hold on
xlabel("BC value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Box Constraint CV")
plot(BCs, BC_cvR2)
savefig( path + "/SoH Results V2/Hyperparams tuning plots/BC Crossval R2.fig")

figure()
hold on
xlabel("log10 KS value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Kernel Scale CV")
plot(log10(KSs), KS_cvR2)
savefig( path + "/SoH Results V2/Hyperparams tuning plots/Log10 KS Crossval R2.fig")

figure()
hold on
xlabel("KS value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Kernel Scale CV")
plot(KSs, KS_cvR2)
savefig( path + "/SoH Results V2/Hyperparams tuning plots/KS Crossval R2.fig")

figure()
hold on
xlabel("log10 Eps value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Eps CV")
plot(log10(Epss), Eps_cvR2)
savefig( path + "/SoH Results V2/Hyperparams tuning plots/Log10 Epsilon Crossval R2.fig")

figure()
hold on
xlabel("Eps value",'FontSize',18 );
ylabel("R2",'FontSize',18 );
title("Eps CV")
plot(Epss, Eps_cvR2)
savefig( path + "/SoH Results V2/Hyperparams tuning plots/Epsilon Crossval R2.fig")



%% Add 1 more CV for the final optimized model, and save all the 5 R2 resulting
finalCV = KFoldCV(5, Xtrain, Ytrain, models{index}, BCs(BCindex), Epss(Epsindex), KSs(KSindex), Kernel, true)
%finalCV = KFoldCV(5, Xtrain, Ytrain, models{3}, BC, Eps, KS, Kernel, true)
%model = fitrsvm(X_array_train(:,models{index}),Y_array_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true);

%% Test model on test set, save the R2 and figs
for i=1:size(Xtest,1)
    figure()
    hold on;
    legend('location', 'best','FontSize',16);
    xlabel('# cycle','FontSize',20 );
    ylabel('SoH','FontSize',20 );
    %title ('Features PCC         N° ' + string(i) +  '       Length: ' + string(size(X{i},1)+"        Policy: "+policy{i}), 'FontSize', 15);
    Test_R2(i) = loss(model,Xtest{i}(:,models{index}), Ytest{i}, 'LossFun', @Rsquared);
   
%     %R2_train = loss(model, XTrain(:,[7 15]), YTrain, 'LossFun', @Rsquared);
    pred = predict(model,Xtest{i}(:,models{index}));
    plot([1:length(Ytest{i})], Ytest{i},  'color','#115eed', 'LineWidth',2,'DisplayName', "Measured "); %+ num2str(R2_train));
    plot ([1:length(Ytest{i})], pred,'.-', 'color','#e87917', 'DisplayName', "Estimated ");
    title ('Test sample '+string(i)+  '       Life cycles: ' + string(size(Xtest{i},1))+ "    R^2: "+ num2str(Test_R2(i)) + "       window: "+dashedWindow,'FontSize',15 );
    savefig( path + "/SoH Results V2/Test plots/Test sample" + int2str(i)+ ".fig");

end
    
disp("Average R2 for the test set is  " + num2str(mean(Test_R2)));


%% Write on file: model, model CV R2, hyperparm values, final CV R2 values

fid=fopen(path + '/SoH Results V2/Results.txt','w');

fprintf(fid, "PARTIAL WINDOW "+ dashedWindow + '\n\n');

fprintf(fid, "Feature 1: Variance    Feature 2: Minimum    Feature 3: Temperature "+  '\n\n');
fprintf(fid, "Selected model (features): "  + int2str(models{index})+  '\n\n');

fprintf(fid, "Model  " + int2str(models{1})  + "  CrossValidation R2: "  + num2str(model_cvR2(1)) + '\n');
fprintf(fid, "Model  " + int2str(models{2})  + "  CrossValidation R2: "  + num2str(model_cvR2(2)) + '\n');
fprintf(fid, "Model  " + int2str(models{3})  + "  CrossValidation R2: "  + num2str(model_cvR2(3)) + '\n\n');

fprintf(fid, "Box Constraint value: " + num2str(BCs(BCindex)) + '\n');
fprintf(fid, "Kernel Scale value: " + num2str(KSs(KSindex)) + '\n');
fprintf(fid, "Epsilon value: " + num2str(Epss(Epsindex)) + '\n\n');

fprintf(fid, "Final model average CrossVal R2: " + num2str(finalCV(1)) + '\n');
fprintf(fid, "Final model single fold CrossVal R2s: " + num2str(finalCV(2:end)) + '\n\n');

fprintf(fid, "Test samples average R2:    "+ num2str(mean(Test_R2))+ '\n');
fprintf(fid, "Singe test samples R2 values:    "+ num2str(Test_R2)+ '\n');

fclose(fid)
