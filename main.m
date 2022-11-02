%% This file contains just a couple of USELESS experiments.
clear all
clc
load('B0005.mat')
load('B0006.mat')
load('B0007.mat')
load("I_8_20_v2.mat")
load("I_25_02_v2.mat")
%load('B00018.mat')


%% Extract the features dataset
batt = B0005;
start_range1 = 3.6;
end_range1 = 4.1;

start_range2 = 4;
end_range2 = 4.2;

step = 0.01;

[X1, Y1] = ExtractPartialCurve(batt,start_range1,step,end_range1);
[X2, Y2] = ExtractPartialCurve(batt,start_range2,step,end_range2);

%%
Y1=Y1/Y1(1)
figure()
xlabel('Cycle #','FontSize',18 );
ylabel('SoH','FontSize',18 );


hold on
plot(Y1, LineWidth=1.4)

%% Plot the data over their measured voltage 
clc
close all
[charge_indices, discharge_indices] = ExtractCyclesIndices(batt);

charging_cycles = cell(1,168);
charging_times = cell(1,168);
discharging_cycles = cell(1,168);
discharging_times = cell(1,168);


for i=1:length(charge_indices)
    charging_cycles{i}  =  batt.cycle(charge_indices(i)).data.Voltage_measured;
    charging_times{i} = batt.cycle(charge_indices(i)).data.Time;

    discharging_cycles{i}  =  batt.cycle(discharge_indices(i)).data.Voltage_measured;
    discharging_times{i} = batt.cycle(discharge_indices(i)).data.Time;
end




figure()
xlabel('Time','FontSize',18 );
ylabel('Voltage','FontSize',18 );


hold on
xlim([0 3600])
ylim([3.4 4.25])

j=0;
colori = ["#ebd534", "#ebc934", "#ebbd34", "#ebab34", "#eb9634", "#eb7d34", "#eb6834" ];
for i=1:25:length(charge_indices)
    %pause(0.2);
    j=j+1;
    plot(charging_times{i}(1:end), charging_cycles{i}(1:end), "DisplayName", "Cycle " +int2str(i), "Color", colori(j), LineWidth=1.4)
    legend('Location','southeast','FontSize',14 );
    legend show
end


%%
figure()
xlabel('Time') 
ylabel('Voltage')
title ('Discharge curves')

hold on

for i=1:1:length(charge_indices)
    %pause(0.2);
    plot(discharging_times{i}(1:end), discharging_cycles{i}(1:end), "DisplayName", int2str(i))
    legend show
end

%% Plot the partial curve over their measured voltage 
%clear all
clc
load('B0005.mat')
load('B0006.mat')
load('B0007.mat')
load("I_8_20_v2.mat")
load("I_25_02_v2.mat")
%load('B00018.mat')

clc
start_range1 = 3.8;
end_range1 = 4;
step = 0.05;

batt =B0005;%I_25_02_v2;% I_8_20_v2;%B0005; 




[X1, Y1] = ExtractPartialCurve(batt,start_range1,step,end_range1);
nom_capacity = batt.cycle(2).data.Capacity;
Y1 = Y1/nom_capacity;


[charge_indices, discharge_indices] = ExtractCyclesIndices(batt);
charging_cycles = cell(1,length(charge_indices));
charging_times = cell(1,length(charge_indices));

for i=1:length(charge_indices)
    full_cycle_voltage = batt.cycle(charge_indices(i)).data.Voltage_measured(3:end);
    full_cycle_time = batt.cycle(charge_indices(i)).data.Time(3:end);

    %first1 = find(full_cycle_voltage <= start_range1, 1, "last");
    first1 = find(full_cycle_voltage>=start_range1,1);
    last1 = find(full_cycle_voltage>=end_range1,1);

    charging_cycles1{i} = full_cycle_voltage(first1:last1);
    charging_times1{i} = full_cycle_time(first1:last1);

end


figure()
hold on
xlabel('Time','FontSize',18 );
ylabel('Voltage','FontSize',18 );




for i=1:1:length(charge_indices)
    %pause(0.05);
    plot(charging_times1{i}, charging_cycles1{i}, "DisplayName", int2str(i))
    %legend show
end




%{
%% PCA
close;
clc
a = [X2 transpose(Y2)];
[coeffs,score,~,~,expl] = pca(a);
pareto(expl);
%}

