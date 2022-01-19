%% Etracting time of each full cycle from B0005
clear all
clc
load('B0005.mat');
load('Battery5.mat');

[ch,dis, imp] = ExtractCyclesIndices(B0005);
charge_times = [];
discharge_times = [];
times = [];

for i=1:length(ch)
    end_charge = find(B0005.cycle(ch(i)).data.Voltage_measured>4.2, 1);
    end_discharge = find(B0005.cycle(dis(i)).data.Voltage_measured<2.7, 1);
    charge_times(i) = B0005.cycle(ch(i)).data.Time(end_charge);
    discharge_times(i) = B0005.cycle(dis(i)).data.Time(end_discharge);
end

times = charge_times+discharge_times;
%%
figure()
plot(1:166, charge_times);
figure()
plot(1:166, discharge_times);
figure()
plot(1:166, times);
%%
k=20;
plot(1:length(B0005.cycle(ch(k)).data.Time),B0005.cycle(ch(k)).data.Time);
hold on
plot(1:length(B0005.cycle(dis(k)).data.Time),B0005.cycle(dis(k)).data.Time, 'r');
%%
