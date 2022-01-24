%% Computing Total Moved Charge from battery data.
%Loading data

close all
clear all
clc

load('B0005.mat');
load('Battery5.mat');

clc
ncyc = 40;

movedc = [ncyc];
total_movedc = [ncyc];

for i=1:ncyc
    cycle = B0005.cycle(i).data;
    times = [0 diff(cycle.Time)];

    %{
    if isfield(cycle, 'Current_charge')
        current = cycle.Current_charge;
    else current = cycle.Current_load;
    end
    %}

    current = abs(cycle.Current_measured);
    movedc(i) = abs(times*transpose(current));
    movedc(i) = movedc(i)/3600;
    
end

for i=1:ncyc/2
    total_movedc(i) = sum(movedc(1:2*i));
end




error_from_Q5 = total_movedc - Q5(1:ncyc/2);
error_derivative = diff(error_from_Q5);
total_movedc = vertcat(total_movedc, Q5(1:ncyc/2));




