%Use this to compute the Inf and Sup of the voltage range for
%extraction. Inf is the maximun among the min of all cycles. Sup is the
%minimum among the maximum of all cycles.

function [inf, sup] = ExtractMaxRange(dataset, charge_indices)
    mins_set = [];
    maxs_set = [];

    for i = charge_indices
       %Extract the entire cycle voltages and times, cutting out first 2 values because are "noise"
       full_cycle_voltage = dataset.cycle(i).data.Voltage_measured(3:end);

       mins_set = [mins_set min(full_cycle_voltage)];
       maxs_set = [maxs_set max(full_cycle_voltage)];
    end
    %mins_set
    inf = max(mins_set);
    sup = min(maxs_set);
end