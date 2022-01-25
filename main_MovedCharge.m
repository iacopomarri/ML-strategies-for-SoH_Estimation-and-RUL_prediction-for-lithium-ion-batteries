%% Computing Total Moved Charge from battery data.
%Loading data

close all
clear all
clc

load('B0005.mat');
load('Battery5.mat');

[ch,dis] = ExtractCyclesIndices(B0005);

movedc = [];
total_movedc = [];

for i=1:length(ch)
    ch_cycle = B0005.cycle(ch(i)).data;
    dis_cycle = B0005.cycle(dis(i)).data;

    ch_times = [0 diff(ch_cycle.Time)];
    dis_times = [0 diff(dis_cycle.Time)];

    %{
    if isfield(cycle, 'Current_charge')
        current = cycle.Current_charge;
    else current = cycle.Current_load;
    end
    %}

    ch_current = abs(ch_cycle.Current_measured);
    dis_current = abs(dis_cycle.Current_measured);

    movedc = [movedc abs(ch_times*transpose(ch_current) + dis_times*transpose(dis_current))/3600];
    total_movedc = [total_movedc sum(movedc(1:i))];
end

%total_movedc = total_movedc/3600;






