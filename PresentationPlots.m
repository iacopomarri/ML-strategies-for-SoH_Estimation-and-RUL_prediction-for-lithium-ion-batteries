
clearvars  -except d
clc

% if exist('d','var')==0
%     d = MITLoadingCode();
% end
 
load('B0005.mat');

%Build datasets from batteries

[~, Y_5] = ExtractPartialCurve(B0005,3.6,0.1,4.1);
nom_capacity5 = B0005.cycle(2).data.Capacity;
Y_5 = Y_5/nom_capacity5;



%%  PLOT for RUL

figure() 
hold on
xlabel('# cycle','FontSize',18 );
ylabel('Ah','FontSize',18 );
yline(0.75, '--', 'Color','r', 'LineWidth',1.5 , "Label", "EoL",'FontSize',14 );

last_cycle = find(Y_5 < 0.75 ,1);

xline(last_cycle, '--', 'Color','r', 'LineWidth',1.5 );%, "Label", "Last useful cycle",'FontSize',18 );
%line([last_cycle last_cycle], [0.65 Y_5(last_cycle)], 'Color','r','Label','Last');
% plot(d(10).summary.QDischarge)
plot(Y_5)


%% PLOT for chargine curves and partial charging curves

