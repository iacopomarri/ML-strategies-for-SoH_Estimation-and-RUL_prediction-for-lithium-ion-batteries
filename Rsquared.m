%% Loss function computing Rsquared

function lossvalue = Rsquared(Y,Yfit,W)
    
    TSS=sum((Y - mean(Y)).^2);
    
    ResiduiL=(Y - Yfit);
    RSS_L=sum(ResiduiL.^2);
    lossvalue= 1-RSS_L/TSS;
end