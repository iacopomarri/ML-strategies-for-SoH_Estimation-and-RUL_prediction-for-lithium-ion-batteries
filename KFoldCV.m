%% K-Fold Cross-validation

%function mean_cross_valid_R2 = KFoldCV(k, Xtrain, Ytrain, ftr_idx, BC, Eps, KS, Kernel)
function Cross_valid_R2 = KFoldCV(k, Xtrain, Ytrain, ftr_idx, BC, Eps, KS, Kernel)
    
    K = k;
    Cross_valid_R2 = [];
    validation_size = floor(size(Xtrain,1)/K);
    
    for i=1:K
        disp("running for K = " + int2str(i) + " ...");
        
        X_crossval_train = cell(size(Xtrain,1), 1);
        Y_crossval_train = cell(size(Xtrain,1), 1);
        X_crossval_valid = [];
        Y_crossval_valid = [];
    
        % copy the full train set
        for j=1:size(Xtrain,1)
            X_crossval_train{j,1} = Xtrain{j}(:, ftr_idx);
            Y_crossval_train{j,1} = Ytrain{j};
        end
    
        % pick validation samples from train set
        for j=validation_size*(i-1)+1:validation_size*i
            X_crossval_valid = vertcat(X_crossval_valid, X_crossval_train{j});
            Y_crossval_valid = [Y_crossval_valid Y_crossval_train{j}];
        end
     
        % delete validation samples from train set
        X_crossval_train(validation_size*(i-1)+1:validation_size*i) = [];
        Y_crossval_train(validation_size*(i-1)+1:validation_size*i) = [];
    
        % transform remaining train data into array format
        Xtemp = [];
        Ytemp = [];
        for j=1:size(X_crossval_train,1)
            Xtemp = vertcat(Xtemp, X_crossval_train{j});
            Ytemp = [Ytemp Y_crossval_train{j}];
        end
    
        X_crossval_train = Xtemp;
        Y_crossval_train = Ytemp;
        clear Xtemp Ytemp;
     
         cross_model = fitrsvm(X_crossval_train, Y_crossval_train,...
             BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel, Standardize=true);
     
         Cross_valid_R2(i) = loss(cross_model,X_crossval_valid, Y_crossval_valid, 'LossFun', @Rsquared);
    end

    %Return value
    Cross_valid_R2 = [mean(Cross_valid_R2) Cross_valid_R2];
   %mean_cross_valid_R2 = mean(Cross_valid_R2);
end