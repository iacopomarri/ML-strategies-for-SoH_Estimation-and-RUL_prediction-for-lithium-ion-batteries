%% SoH Estimation. SVM trained on 5-6-7, tested on 18
clear all
clc
load('B0005.mat');
load('B0006.mat');
load('B0007.mat');
load('B0018.mat');

% Measure nominal capacities
nom_capacity5 = B0005.cycle(2).data.Capacity;
nom_capacity6 = B0006.cycle(2).data.Capacity;
nom_capacity7 = B0007.cycle(2).data.Capacity;
nom_capacity18 = B0018.cycle(3).data.Capacity;

%[X_train, Y_train] = ExtractPartialCurve(B0005,3.8,0.05,4);
%% Hyperparameters tuning test: picking values to try
clc

start_ = 3.7:0.1:3.9;
width_ = 0.2:0.1:0.2;
%step_ = 0.005:0.005:0.2;

err_array = [];
err_matrix = [];
err_cube = [];
%% Hyperparameters tuning test: procedure

%With k we change the width of the voltage window for our partial charging curve
for k=1:length(width_)
  
    end_ = start_ + width_(k);
    step_ = (width_(k)/20):(width_(k)/20):width_(k);
    err_matrix = zeros(length(start_), length(step_));

    %With i we translate the window for partial ch. curve from 3.6 to 4.2 V
    for i=1:length(end_)

        %With j we change the length of the timesteps we measure
        for j=1:length(step_)
  
            err = 0;
            [X_5, Y_5] = ExtractPartialCurve(B0005,start_(i),step_(j),end_(i));
            [X_6, Y_6] = ExtractPartialCurve(B0006,start_(i),step_(j),end_(i));
            [X_7, Y_7] = ExtractPartialCurve(B0007,start_(i),step_(j),end_(i));
            %[X_18, Y_18] = ExtractPartialCurve(B0018,start_(i),step_(j),end_(i));
            
            Y_5 = Y_5/nom_capacity5;
            Y_6 = Y_6/nom_capacity6;
            Y_7 = Y_7/nom_capacity7;
           % Y_18 = Y_18/nom_capacity18;
            


            %We compute the error performing a LOOCV between all batteries we have
            X_train = vertcat(X_6, X_7);
            Y_train = [Y_6 Y_7];
            X_test = X_5;
            Y_test = Y_5;
            %Fit model, compute prediction and error
            model = fitrsvm(X_train,Y_train);
            err = err + loss(model,X_test, Y_test, 'LossFun', @Rsquared);

            X_train = vertcat(X_5, X_7);
            Y_train = [Y_5 Y_7];
            X_test = X_6;
            Y_test = Y_6;
            %Fit model, compute prediction and error
            model = fitrsvm(X_train,Y_train);
            err = err + loss(model,X_test, Y_test, 'LossFun', @Rsquared);

            X_train = vertcat(X_5, X_6);
            Y_train = [Y_5 Y_6];
            X_test = X_7;
            Y_test = Y_7;
            %Fit model, compute prediction and error
            model = fitrsvm(X_train,Y_train);
            err = err + loss(model,X_test, Y_test, 'LossFun', @Rsquared);
    
            err = err/3;
    
          
            err_array(j) = err;
            print = ['Test N° ' num2str(k) ' , '  num2str(i) ' , '  num2str(j) ' done, R2 = ' num2str(err)];
            disp(print);
        end
    
        err_matrix(i,:) = err_array;
       % err_array = [];
    end
    
    err_cube(:,:,k) = err_matrix ;
    %err_matrix = [];
end

%% Find optimal model first test
%err_cube = load('tuning_test_R2_1.mat').err_cube;

[massimo1, massimo1_indice]  = max(err_cube);
[massimo2, massimo2_indice] = max(massimo1);
[massimo, massimo_indice] = max(massimo2)

find(err_cube(:,:,:) == massimo)


%ampiezza 0,2V, step_size = 0.022, range = 4.0 - 4.2 V

%% Plots R2 results first test
%err_cube = load('tuning_test_R2_1.mat').err_cube;

start_ = 3.4:0.1:4;
width_ = 0.2:0.1:0.6;
step_ = 0.001:0.003:0.1;

for k=1:length(width_)
    err = err_cube(:,:,k); %load 1 of k matrices for the plot
    avg = mean(transpose(err));
    a = find(err(:,1)==0); %find any zeros row 
    err(a,:) = [];         %remove those rows
    err=transpose(err);    %transpose so that steps size goes on X axis

    figure();
    plot(step_, err);
    yline(massimo, '-', 'color','#090a0a', 'LineWidth',1.5);
    lgd = legend(string(start_) + ' - ' + string(start_+width_(k)) + ' V' + '  /  AVG R^2: ' + avg, Location="southeast");
    xlabel('Size of sampling steps (V)') ;
    ylabel('R^2');
    title ('B0018 estimation with SVM, partial curves width = ' + string(width_(k)));
end

%% Find optimal model second test
%err_cube = load('tuning_test_R2_2.mat').err_cube;

[massimo1, massimo1_indice]  = max(err_cube);
[massimo2, massimo2_indice] = max(massimo1);
[massimo, massimo_indice] = max(massimo2)

find(err_cube(:,:,:) == massimo)

%% Plots R2 results second test
%err_cube = load('tuning_test_R2_2.mat').err_cube;

start_ = 3.7:0.1:3.9;
width_ = 0.2:0.1:0.2;

for k=1:length(width_)
    step_ = (width_(k)/20):(width_(k)/20):width_(k);
    
    R2 = err_cube(:,:,k); %load 1 of k matrices for the plot
    avg = mean(transpose(R2));
    a = find(R2(:,1)==0); %find any zeros row 
    R2(a,:) = [];         %remove those rows
    R2=transpose(R2);    %transpose so that steps size goes on X axis

    figure();
    plot(step_, R2);
    yline(massimo, '-', 'color','#090a0a', 'LineWidth',1.5);
    lgd = legend(string(start_) + ' - ' + string(start_+width_(k)) + ' V' + '  /  AVG R^2: ' + avg, Location="southeast");
    xlabel('Size of sampling steps (V)') ;
    ylabel('R^2');
    title ('B0018 estimation with SVM, partial curves width = ' + string(width_(k)));
end


%% Find optimal model 
[massimo1, massimo1_indice]  = max(err_cube);
[massimo2, massimo2_indice] = max(massimo1);
[massimo, massimo_indice] = max(massimo2)

find(err_cube(:,:,:) == massimo)

%% Training the optimal model found before




