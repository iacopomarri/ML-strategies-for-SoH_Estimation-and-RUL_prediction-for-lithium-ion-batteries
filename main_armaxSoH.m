close all
clearvars  -except d
clc


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

%Build datasets from batteries
start_range1 = 3.9; %3.9;
end_range1 = 4; %4;
step = 0.05;%0.1;

%{
X_5 = [X_5 ExtractTotalMovedCharge(B0005)];
X_6 = [X_6 ExtractTotalMovedCharge(B0006)];
X_7 = [X_7 ExtractTotalMovedCharge(B0007)];
X_18 = [X_18 ExtractTotalMovedCharge(B0018)];

X_5 = ExtractTotalMovedCharge(B0005);
X_6 = ExtractTotalMovedCharge(B0006);
X_7 = ExtractTotalMovedCharge(B0007);
X_18 = ExtractTotalMovedCharge(B0018);
%}

[X_5, Y_5] = ExtractPartialCurve(B0005,start_range1,step,end_range1);
[X_6, Y_6] = ExtractPartialCurve(B0006,start_range1,step,end_range1);
[X_7, Y_7] = ExtractPartialCurve(B0007,start_range1,step,end_range1);
[X_18, Y_18] = ExtractPartialCurve(B0018,start_range1,step,end_range1);

Y_5 = Y_5/nom_capacity5;
Y_6 = Y_6/nom_capacity6;
Y_7 = Y_7/nom_capacity7;
Y_18 = Y_18/nom_capacity18;


% seems that adding more than 2 features decreases the performances. If we
% had more than 3 data we coould handle more complexity

% X_5 = [X_5 ExtractTotalMovedCharge(B0005)];
% X_6 = [X_6 ExtractTotalMovedCharge(B0006)];
% X_7 = [X_7 ExtractTotalMovedCharge(B0007)];
% X_18 = [X_18 ExtractTotalMovedCharge(B0018)];




%%
%load PredPreyCrowdingData
y=Y_5';
z = iddata(y,[],1,'TimeUnit','minutes','Tstart',0);
plot(z)
title('Predator-Prey Population Data')
ylabel('SoH')
%%
estlim = 120;
ze = z(1:estlim);
zv = z(estlim+1:end);
[zed, Tze] = detrend(ze, 0);
[zvd, Tzv] = detrend(zv, 0);
na_list = (1:10)';
V1 = arxstruc(zed(:,1,:),zvd(:,1,:),na_list);
na1 = selstruc(V1,0);


% V2 = arxstruc(zed(:,2,:),zvd(:,2,:),na_list);
% na2 = selstruc(V2,0);

na = [na1 na1-1];

nc = [];

%%
sysARMA = armax(zed,[na nc]) 
%%

predOpt = predictOptions('OutputOffset',Tze.OutputOffset');
yhat1 = predict(sysARMA,ze,5, predOpt);
plot(ze,yhat1)
% compareOpt = compareOptions('OutputOffset',Tze.OutputOffset');
% compare(ze,sysARMA,10,compareOpt)

forecastOpt = forecastOptions('OutputOffset',Tze.OutputOffset');
[yf1,x01,sysf1,ysd1] = forecast(sysARMA, ze, 45, forecastOpt);
t = yf1.SamplingInstants;
te = ze.SamplingInstants;
t0 = z.SamplingInstants;
% subplot(1,2,1);
plot(t0,z.y(:,1),...
   te,yhat1.y(:,1),...
   t,yf1.y(:,1),'m',...
   t,yf1.y(:,1)+ysd1(:,1),'k--', ...
   t,yf1.y(:,1)-ysd1(:,1), 'k--')
    title ('Armax process, B0005 ', 'FontSize',18 ); 
  xlabel('N° of cycle','FontSize',18 );
    ylabel('SoH','FontSize',18 );
    legend({'Measured','Fitted','Predicted estimation','Prediction Uncertainty (1 sd)'},...
   'Location','best','FontSize',12)
