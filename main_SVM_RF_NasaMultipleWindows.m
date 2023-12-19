%% SVR-RF experiments on multiple voltage windows.
% Setting up datasets
close all
clearvars  -except d
clc

rng(10);
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


%windows done
% 3.8 3.85;
%     3.85 3.9;
%     3.9 3.95;

windows = [
    3.95 4;
    3.8 3.9;
    3.85 3.95;
    3.9 4;
    3.75 3.9;
    3.8 3.95;
    3.85 4;
    3.9 4.05;
    3.8 4;
    3.85 4.05;
    3.9 4.1
   ]

windows = [
    3.9 4
   ]


for i=1:size(windows,1)
    numWindow = windows(i,:)

    %Build datasets from batteries
    %numWindow = [3.9 4];  %change this to perform the process on a different window
    start_range1 = numWindow(1); 
    end_range1 = numWindow(2);  
    step = 0.05;
    
    stringStep = num2str(step)
    stringWindow = num2str(numWindow(1) + "_" +num2str(numWindow(2)) + " V");
    dashedWindow = num2str(numWindow(1) + "-" +num2str(numWindow(2)) + " V");
    
    
    [X_5, Y_5] = ExtractPartialCurve(B0005,start_range1,step,end_range1);
    [X_6, Y_6] = ExtractPartialCurve(B0006,start_range1,step,end_range1);
    [X_7, Y_7] = ExtractPartialCurve(B0007,start_range1,step,end_range1);
    [X_18, Y_18] = ExtractPartialCurve(B0018,start_range1,step,end_range1);
    
    Y_5 = Y_5/nom_capacity5;
    Y_6 = Y_6/nom_capacity6;
    Y_7 = Y_7/nom_capacity7;
    Y_18 = Y_18/nom_capacity18;

    X = cell(3,1);
    Y = cell(3,1);
    X{1} = X_5;
    X{2} = X_6;
    X{3} = X_7;
    Y{1} = Y_5';
    Y{2} = Y_6';
    Y{3} = Y_7';
    
    %SVM hyperparams
    BC =  0.1989;      % >0.2 ok
    KS = 11.55;   
    Eps = 0.03013;%0.030107; %0.00128
    Kernel = 'linear';
    
    % Optimize parameters for all 3 batteries (if fit() is commented is because in the next section, best found parameters are hardcoded and used).
    %you may not need to run this section since best parameters are already
    %hardcoded
    % Models, Results and plots
    optimize = false
    if optimize
        X_train = vertcat(X_5, X_6, X_7);
        Y_train = [Y_5 Y_6 Y_7];
        
        rng default
        hyperpar = ["BoxConstraint", "KernelScale", "Epsilon"];
        model_1 = fitrsvm(X_train,Y_train,  "OptimizeHyperparameters",hyperpar, "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=100));
        BC = 0.001052; %model_1.ModelParameters.BoxConstraint;       
        KS = 21.344; %model_1.ModelParameters.KernelScale;
        Eps = 0.00128; %model_1.ModelParameters.Epsilon;   
        Kernel = 'linear'; % model_1.ModelParameters.KernelFunction;
         % 0.016175         480.97       0.00014739
    end
    
    % Build directories
    path ="../../SoH Nasa experiments/" + stringWindow +"/Step "+stringStep;
    
    if ~exist(path + "/SoH Nasa Results", 'dir')
           mkdir(path + "/SoH Nasa Results");
    end
    
    if ~exist(path + "/SoH Nasa Results/Loocv matlab data", 'dir')
       mkdir(path + "/SoH Nasa Results/Loocv matlab data");
    end
    
    if ~exist(path + "/SoH Nasa Results/Test plots", 'dir')
       mkdir(path + "/SoH Nasa Results/Test plots");
    end
    
    if ~exist(path + "/SoH Nasa Results/Hyperparams tuning plots", 'dir')            
       mkdir(path + "/SoH Nasa Results/Hyperparams tuning plots");
    end
    


    SVM = false
    RF = ~SVM
 
    %As a first try, explore the log space (0.001, 0.01, 0.1, 1, 10, etc)
    % logspaceProbing = false
    % if logspaceProbing
    %     multiplcationFactors = logspace(-3, 3, 3);
    %     BCs = multiplcationFactors;
    %     KSs = multiplcationFactors;
    %     Epss = multiplcationFactors;
    % end
    
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


    %FOR RF
    numTrees = 1:50;
    maxSplits = 1:50;

    NT_cvR2 = [];
    MS_cvR2 = [];
    

  %INCLUDERE QUESTO CODICE IN UNA SEZIONE ED ESEGUIRLO, PER VISUALIZZARE GLI ALBERI DECISIONALI 
   X_train = X;
   X_train(2) = []; %rmove B6
   Y_train = Y;
   Y_train(2) = []; %remove B6
    
    X_train = cat(1, X_train{:});
    Y_train = cat(1, Y_train{:});

   model = TreeBagger(2, X_train, Y_train, 'Method', 'regression', 'MaxNumSplits', 100, 'InBagFraction',0.33);
   
   view(model.Trees{1},Mode="graph")

   view(model.Trees{2},Mode="graph")
 %%

    tStart = tic
    parfor i=1:size(numTrees,2)
        tic
        fprintf('"Pyramidal cv, window %.2f', numWindow(1));
        fprintf('-%.2f', numWindow(2))
        %fprintf(' running BC for i = %i', i);
        fprintf(' running NT for i = %i', i);
        fprintf('...');
       
        results = LooCV_RF(X, Y, numTrees(i), 100, false);  
    
        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
       
        %BC_cvR2(i) = results(1); 
        NT_cvR2(i) = results(1); 
    end
    %[~, BCindex] =  max(BC_cvR2);
    [~, NTindex] =  max(NT_cvR2);
    
    

    parfor i=1:size(maxSplits,2)
        tic
        fprintf('"Pyramidal cv, window %.2f', numWindow(1));
        fprintf('-%.2f', numWindow(2))
       % fprintf(' running KS for i = %i', i);
        fprintf(' running MS for i = %i', i);
        fprintf('...');
    
        %results = LooCV_GPT(X, Y, BCs(BCindex), Eps, KSs(i), Kernel, false);
        results = LooCV_RF(X, Y, numTrees(NTindex), maxSplits(i), false);  

        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
      
        %KS_cvR2(i) = results(1);
        MS_cvR2(i) = results(1);
    end
    [~, MSindex] =  max(MS_cvR2);

    tEnd = toc(tStart);
    disp(['Total time: ' num2str(tEnd) ' s'])
%     for i=1:size(EpsmultiplcationFactors,2)
%         tic
%         fprintf('"Pyramidal cv, window %.2f', numWindow(1));
%         fprintf('-%.2f', numWindow(2))
%         fprintf(' running Eps for i = %i', i);
%         fprintf('...');
%     
%         results = LooCV_GPT(X, Y, BCs(BCindex), Epss(i), KSs(KSindex), Kernel, false);
%     
%         fprintf('   ET: %f', toc);    % int2str(i)
%         fprintf(' sec \n');
%         total_time = total_time + toc;
%         Eps_cvR2(i) = results(1);
%     end
%     [~, Epsindex] =  max(Eps_cvR2);


    
   
    
    %save(path + "/SoH Nasa Results/Loocv matlab data/pyramid_SVM_cv_R2_" + dashedWindow +".mat", "BC_cvR2", "KS_cvR2", "Eps_cvR2");
    
%     [~, BCindex] =  max(BC_cvR2);
%     [~, KSindex] =  max(KS_cvR2);
%     [~, Epsindex] =  max(Eps_cvR2);

    
    % Add 1 more CV for the final optimized model, and save all the 5 R2 resulting
    %finalCV = LooCV(X, Y, BCs(BCindex), Epss(Epsindex), KSs(KSindex), Kernel, true);
    finalCV = LooCV_RF(X, Y, numTrees(NTindex), maxSplits(MSindex), false);
    fprintf('"Final LooCV R2 is %.4f', finalCV(1));
        fprintf('\n');


    
    % Write on file: model, model CV R2, hyperparm values, final CV R2 values
    write = true
    
    if write
        if SVM
            fid=fopen(path + '/SoH Nasa Results/SVM_Results.txt','w');
            
            fprintf(fid, "PARTIAL WINDOW "+ dashedWindow + '\n\n');
            
            fprintf(fid, "Box Constraint value: " + num2str(BCs(BCindex)) + '\n');
            fprintf(fid, "Kernel Scale value: " + num2str(KSs(KSindex)) + '\n');
            fprintf(fid, "Epsilon value: " + num2str(Epss(Epsindex)) + '\n\n');
            
            fprintf(fid, "Final model average CrossVal R2: " + num2str(finalCV(1)) + '\n');
            fprintf(fid, "Final model single fold CrossVal R2s: " + num2str(finalCV(2:end)) + '\n\n');
            
            fclose(fid)

        elseif RF
             fid=fopen(path + '/SoH Nasa Results/RF_Results.txt','w');
            
            fprintf(fid, "PARTIAL WINDOW "+ dashedWindow + '\n\n');
            
            fprintf(fid, "Number of trees: " + num2str(numTrees(NTindex)) + '\n');
            fprintf(fid, "Max splits: " + num2str(maxSplits(MSindex)) + '\n');
            
            fprintf(fid, "Final model average CrossVal R2: " + num2str(finalCV(1)) + '\n');
            fprintf(fid, "Final model single fold CrossVal R2s: " + num2str(finalCV(2:end)) + '\n\n');
        end
    end
end



maxSplits(MSindex)
numTrees(NTindex)








%% SECOND PHASE ---- DELETE WHEN DONE










%% SVR-RF experiments on multiple voltage windows.
% Setting up datasets
close all
clearvars  -except d
clc

rng(10);
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

%%
windows = [3.7 3.8;
    3.8 3.9;
    3.9 4;
    4 4.1;
    3.7 3.9; 
    3.8 4;
    3.9 4.1    
   ]

windows = [
    3.9 4;
    3.8 4
   ]

%%

for i=1:size(windows,1)
    numWindow = windows(i,:)

    %Build datasets from batteries
    %numWindow = [3.9 4];  %change this to perform the process on a different window
    start_range1 = numWindow(1); 
    end_range1 = numWindow(2);  
    step = 0.1;
    
    stringStep = num2str(step)
    stringWindow = num2str(numWindow(1) + "_" +num2str(numWindow(2)) + " V");
    dashedWindow = num2str(numWindow(1) + "-" +num2str(numWindow(2)) + " V");
    
    
    [X_5, Y_5] = ExtractPartialCurve(B0005,start_range1,step,end_range1);
    [X_6, Y_6] = ExtractPartialCurve(B0006,start_range1,step,end_range1);
    [X_7, Y_7] = ExtractPartialCurve(B0007,start_range1,step,end_range1);
    [X_18, Y_18] = ExtractPartialCurve(B0018,start_range1,step,end_range1);
    
    Y_5 = Y_5/nom_capacity5;
    Y_6 = Y_6/nom_capacity6;
    Y_7 = Y_7/nom_capacity7;
    Y_18 = Y_18/nom_capacity18;
    
    X = cell(3,1);
    Y = cell(3,1);
    X{1} = X_5;
    X{2} = X_6;
    X{3} = X_7;
    Y{1} = Y_5';
    Y{2} = Y_6';
    Y{3} = Y_7';
    
    %SVM hyperparams
    BC =  0.1989;      % >0.2 ok
    KS = 11.55;   
    Eps = 0.03013;%0.030107; %0.00128
    Kernel = 'linear';
    
    % Optimize parameters for all 3 batteries (if fit() is commented is because in the next section, best found parameters are hardcoded and used).
    %you may not need to run this section since best parameters are already
    %hardcoded
    % Models, Results and plots
    optimize = false
    if optimize
        X_train = vertcat(X_5, X_6, X_7);
        Y_train = [Y_5 Y_6 Y_7];
        
        rng default
        hyperpar = ["BoxConstraint", "KernelScale", "Epsilon"];
        model_1 = fitrsvm(X_train,Y_train,  "OptimizeHyperparameters",hyperpar, "HyperparameterOptimizationOptions", struct(MaxObjectiveEvaluations=100));
        BC = 0.001052; %model_1.ModelParameters.BoxConstraint;       
        KS = 21.344; %model_1.ModelParameters.KernelScale;
        Eps = 0.00128; %model_1.ModelParameters.Epsilon;   
        Kernel = 'linear'; % model_1.ModelParameters.KernelFunction;
         % 0.016175         480.97       0.00014739
    end
    
    % Build directories
    path ="../../SoH Nasa experiments/" + stringWindow +"/Step "+stringStep;
    
    if ~exist(path + "/SoH Nasa Results", 'dir')
           mkdir(path + "/SoH Nasa Results");
    end
    
    if ~exist(path + "/SoH Nasa Results/Loocv matlab data", 'dir')
       mkdir(path + "/SoH Nasa Results/Loocv matlab data");
    end
    
    if ~exist(path + "/SoH Nasa Results/Test plots", 'dir')
       mkdir(path + "/SoH Nasa Results/Test plots");
    end
    
    if ~exist(path + "/SoH Nasa Results/Hyperparams tuning plots", 'dir')            
       mkdir(path + "/SoH Nasa Results/Hyperparams tuning plots");
    end
    


    SVM = false
    RF = ~SVM
 
    %As a first try, explore the log space (0.001, 0.01, 0.1, 1, 10, etc)
    % logspaceProbing = false
    % if logspaceProbing
    %     multiplcationFactors = logspace(-3, 3, 3);
    %     BCs = multiplcationFactors;
    %     KSs = multiplcationFactors;
    %     Epss = multiplcationFactors;
    % end
    
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


    %FOR RF
    numTrees = 1:1000;
    maxSplits = 1:1000;

    NT_cvR2 = [];
    MS_cvR2 = [];
    

   
    tStart = tic
    parfor i=1:size(numTrees,2)
        tic
        fprintf('"Pyramidal cv, window %.2f', numWindow(1));
        fprintf('-%.2f', numWindow(2))
        %fprintf(' running BC for i = %i', i);
        fprintf(' running NT for i = %i', i);
        fprintf('...');
       
        results = LooCV_RF(X, Y, numTrees(i), 100, false);  
    
        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
       
        %BC_cvR2(i) = results(1); 
        NT_cvR2(i) = results(1); 
    end
    %[~, BCindex] =  max(BC_cvR2);
    [~, NTindex] =  max(NT_cvR2);
    
    

    parfor i=1:size(maxSplits,2)
        tic
        fprintf('"Pyramidal cv, window %.2f', numWindow(1));
        fprintf('-%.2f', numWindow(2))
       % fprintf(' running KS for i = %i', i);
        fprintf(' running MS for i = %i', i);
        fprintf('...');
    
        %results = LooCV_GPT(X, Y, BCs(BCindex), Eps, KSs(i), Kernel, false);
        results = LooCV_RF(X, Y, numTrees(NTindex), maxSplits(i), false);  

        fprintf('   ET: %f', toc);    % int2str(i)
        fprintf(' sec \n');
      
        %KS_cvR2(i) = results(1);
        MS_cvR2(i) = results(1);
    end
    [~, MSindex] =  max(MS_cvR2);

    tEnd = toc(tStart);
    disp(['Total time: ' num2str(tEnd) ' s'])
%     for i=1:size(EpsmultiplcationFactors,2)
%         tic
%         fprintf('"Pyramidal cv, window %.2f', numWindow(1));
%         fprintf('-%.2f', numWindow(2))
%         fprintf(' running Eps for i = %i', i);
%         fprintf('...');
%     
%         results = LooCV_GPT(X, Y, BCs(BCindex), Epss(i), KSs(KSindex), Kernel, false);
%     
%         fprintf('   ET: %f', toc);    % int2str(i)
%         fprintf(' sec \n');
%         total_time = total_time + toc;
%         Eps_cvR2(i) = results(1);
%     end
%     [~, Epsindex] =  max(Eps_cvR2);


    
   
    
    %save(path + "/SoH Nasa Results/Loocv matlab data/pyramid_SVM_cv_R2_" + dashedWindow +".mat", "BC_cvR2", "KS_cvR2", "Eps_cvR2");
    
%     [~, BCindex] =  max(BC_cvR2);
%     [~, KSindex] =  max(KS_cvR2);
%     [~, Epsindex] =  max(Eps_cvR2);

    
    % Add 1 more CV for the final optimized model, and save all the 5 R2 resulting
    %finalCV = LooCV(X, Y, BCs(BCindex), Epss(Epsindex), KSs(KSindex), Kernel, true);
    finalCV = LooCV_RF(X, Y, numTrees(NTindex), maxSplits(MSindex), false);
    fprintf('"Final LooCV R2 is %.4f', finalCV(1));
        fprintf('\n');
    
    
    % Write on file: model, model CV R2, hyperparm values, final CV R2 values
    write = true
    
    if write
        if SVM
            fid=fopen(path + '/SoH Nasa Results/SVM_Results.txt','w');
            
            fprintf(fid, "PARTIAL WINDOW "+ dashedWindow + '\n\n');
            
            fprintf(fid, "Box Constraint value: " + num2str(BCs(BCindex)) + '\n');
            fprintf(fid, "Kernel Scale value: " + num2str(KSs(KSindex)) + '\n');
            fprintf(fid, "Epsilon value: " + num2str(Epss(Epsindex)) + '\n\n');
            
            fprintf(fid, "Final model average CrossVal R2: " + num2str(finalCV(1)) + '\n');
            fprintf(fid, "Final model single fold CrossVal R2s: " + num2str(finalCV(2:end)) + '\n\n');
            
            fclose(fid)

        elseif RF
             fid=fopen(path + '/SoH Nasa Results/RF_Results.txt','w');
            
            fprintf(fid, "PARTIAL WINDOW "+ dashedWindow + '\n\n');
            
            fprintf(fid, "Number of trees: " + num2str(numTrees(NTindex)) + '\n');
            fprintf(fid, "Max splits: " + num2str(maxSplits(MSindex)) + '\n');
            
            fprintf(fid, "Final model average CrossVal R2: " + num2str(finalCV(1)) + '\n');
            fprintf(fid, "Final model single fold CrossVal R2s: " + num2str(finalCV(2:end)) + '\n\n');
        end
    end
end


