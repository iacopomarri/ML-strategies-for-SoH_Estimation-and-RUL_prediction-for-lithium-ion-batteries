%% SVR experiments: using different batt from different packs to find those appropriate for teseting.
% Setting up datasets
close all
clearvars  -except d
clc

rng(10);


%% LOAD ALL FILES

num_batt = 34;
path = 'NASA full dataset\';
folder_list = dir(path)
folder_list(~[folder_list.isdir]) = [];  % Keep only the folders
folder_list(ismember({folder_list.name}, {'.', '..'})) = [];  % Remove '.' and '..'
folder_list(2) = []  % Remove folder with bat 25-26-27-28. they're contained in the 3rd folder already
num_folders = length(folder_list);

B = cell(num_batt,4);
k=0;

for i = 1:num_folders
   folder_name = folder_list(i).name;  % Get the name of the current folder
   folder_path = fullfile(path, folder_name);  % Get the full path to the file
   file_list = dir(fullfile(folder_path, '*.mat'))  % List all .mat files in folder
   num_files = length(file_list);  % Get the number of files

   for j = 1:num_files
       k = k+1;
       file_name = file_list(j).name;  % Get the name of the current file
       file_path = fullfile(folder_path, file_name);  % Get the full path to the file
       B{k,1} = file_name(1:5);
       data = load(file_path);
       B{k,2} = data.(file_name(1:5));  % Load the file
       disp(file_path)
       % Do something with the loaded data
   end
end



%% EXTRACT DATA
% 
% window = [3.7 4.15];
% step = 0.05;
window = [4 4.05];
step = 0.05;

% window = [3.9 4];
% step = 0.1;

% window = [3.8 4.1];
% step = 0.1;

% Plot new batteries
plot_all = false;

for i = 1:num_batt
    [B{i,3}, B{i,4}] = ExtractPartialCurve(B{i,2},window(1),step,window(2));

    if plot_all
        figure 
        hold on
        xlabel('Cycle #','FontSize',18 );
        ylabel('Capacity','FontSize',18 );
        title(B{i,1}, 'FontSize',18);
        plot(B{i,4}, LineWidth=1.4)
    end

    B{i,4} = B{i,4} / B{i,4}(2); % Converting capacity into SoH
end

%Remove lines 30 and 28: they have unmatched numbers of X and Y
B(30,:) = [];
B(28,:) = [];
%recompute number of batteries
num_batt = length(B);



% if plot_all
%     for i = 1:num_batt
%         figure 
%         hold on
%         xlabel('Cycle #','FontSize',18 );
%         ylabel('SoH','FontSize',18 );
%         title(B{i,1}, 'FontSize',18);
%         plot(B{i,4}, LineWidth=1.4)
%     end
% end

%clearvars  -except B step window

%% LOAD OPTIMAL MODEL

  %SVM hyperparams
BC =  0.1989;      % >0.2 ok
KS = 11.55;   
Eps = 0.03013;%0.030107; %0.00128
Kernel = 'linear';

%FOR SVR
multiplcationFactors = 0.0001:0.005:5;

BCmultiplcationFactors = [0.0001:0.001:1 1:0.05:3];
KSmultiplcationFactors = 0.0001:0.05:7;
EpsmultiplcationFactors = [0.0001:0.001:1 1:0.05:3];

%Values to test
BCs = BCmultiplcationFactors*BC;
KSs = KSmultiplcationFactors*KS;
Epss = EpsmultiplcationFactors*Eps;

BC_cvR2 = [];
KS_cvR2 = [];
Eps_cvR2 = [];

parametri = load('../../SoH Nasa experiments/3.9_4 V/Step 0.05/SoH Nasa Results/Loocv matlab data/pyramid_SVM_cv_R2_3.9-4 V.mat');
[~,BCindex] = max(parametri.BC_cvR2);
[~, KSindex] = max(parametri.KS_cvR2);
[~, Epsindex] = max(parametri.Eps_cvR2);

BC = BCs(BCindex);
KS = KSs(KSindex); 
Eps = Epss(Epsindex); 
Kernel='linear';
clearvars  -except B step window BC KS Eps Kernel num_batt
%% SMOOTH FEATURES VALUES

for i=1:size(B,1)
    for j=1:size(B{i,3},2)
        B{i,3}(:,j) =  smooth(B{i,3}(:,j))
    end
end

%% TRASLATE B18 FEATURE VALUES LEFT
% tras=1
% 
% for k=1:tras
%     for i=1:size(B{4,3},2)-1
%       B{4,3}(:,i) = B{4,3}(:,i+1)  
%     end
% end


%% TRAIN ON 1,2,3 TEST ON 1, 2, 3 and 4 (4 is 18th)
clc
R2 = []

X_train = vertcat(B{1,3},B{2,3}, B{3,3});
Y_train = vertcat(B{1,4}',B{2,4}', B{3,4}');

X_test = B{4,3};
Y_test = B{4,4};


%Fit model, compute prediction and R2

model = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction='linear', Standardize=false);
R2(i) = loss(model ,X_test, Y_test, 'LossFun', @Rsquared);

fprintf('index %i', i);
fprintf(',  R2 :%f', R2(i));
fprintf(' \n');
    




%% TRAIN ON ALL BATTERIES BUT 1, TEST ON THAT 1. ITERATE OVER ALL BATTERIES
clc
maxBatteries = num_batt;
batteries = [1 2 3 4]; %add 7
Btemp = B([batteries],:);
%Btemp = B([1:maxBatteries],:);
R2 = zeros(size(Btemp,1),1); % Preallocate R2 vector


for i = 1:size(batteries,2)%1:maxBatteries 
    
    % Split data into training and test sets
    X_test = Btemp{i,3};
    Y_test = Btemp{i,4}';
    B_train = Btemp([1:i-1 i+1:end],:); % Exclude current dataset from training set
    for j=1:size(B_train,1)
        B_train{j,4} = B_train{j,4}';
    end
    X_train = vertcat(B_train{:,3});
    Y_train = vertcat(B_train{:,4});
    
    % Train and test support vector regressor
    model = fitrsvm(X_train,Y_train, 'BoxConstraint', BC, 'Epsilon', Eps, 'KernelScale', KS, 'KernelFunction', 'linear');
    R2(i) = loss(model ,X_test, Y_test, 'LossFun', @Rsquared);

    fprintf('Test battery %s', B{batteries(i),1});
    fprintf(':  R2 :%f', R2(i));
    fprintf(' \n');
end

mean(R2(1:end-1))
figure 
hold on
pred_a = predict(model,X_test);
plot([1:length(Y_test)], Y_test, 'r');
plot ([1:length(Y_test)], pred_a,'r--');
legend({"R2: " + string(R2(4)), "limits: " + string(window(1))+"-"+string(window(2))}, 'location', 'best','FontSize',16);

title(string(window) + string(step))
