function R2 = LooCV_SVR(X, Y, BC, Eps, KS, Kernel, Verbose)
    R2 = zeros(1, size(X,1));
    parfor i=1:size(X,1)
        X_train = X;
        X_train(i) = [];
        Y_train = Y;
        Y_train(i) = [];

        X_train = cat(1, X_train{:});
        Y_train = cat(1, Y_train{:});

        %Fit model, compute prediction and R2
     
        model = fitrsvm(X_train,Y_train, BoxConstraint = BC, Epsilon = Eps, KernelScale=KS, KernelFunction=Kernel);
        R2(i) = loss(model ,X{i}, Y{i}, 'LossFun', @Rsquared);
        
    end
    R2 = [mean(R2) R2];
end
