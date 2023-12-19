function R2 = LooCV_RF(X, Y, numTrees, maxNumSplits, Verbose)
  R2 = zeros(1, size(X,1));
  rng(10);
    parfor i=1:size(X,1)
        X_train = X;
        X_train(i) = [];
        Y_train = Y;
        Y_train(i) = [];

        X_train = cat(1, X_train{:});
        Y_train = cat(1, Y_train{:});

        %Fit RF model, compute prediction and R2
        model = TreeBagger(numTrees, X_train, Y_train, 'Method', 'regression', 'MaxNumSplits', maxNumSplits, 'InBagFraction',0.33);
        figure()
        hold on
        view(model.Trees{1},Mode="graph")
        Ypred = predict(model, X{i});
        
        % Valuta le prestazioni del modello
        R2(i) = 1 - sum((Y{i} - Ypred).^2) / sum((Y{i} - mean(Y{i})).^2);



    end
    R2 = [mean(R2) R2];
end
    