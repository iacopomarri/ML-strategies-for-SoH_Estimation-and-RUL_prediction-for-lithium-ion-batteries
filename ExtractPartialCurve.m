%This function is built for NASA Li-Ion datasets B0005, B0006, B0007, B00018

%TODO: extends to other type of feature extraction: eg. relative capacities or other. Add the input "Option".

%INPUTs: dataset, range of values for the partial curve (mV), step length of the sampling procedure.
%OUTPUTs: X: matrix of features arrays. 
%         Y: Target capacity values. 



function [X,Y] = ExtractPartialCurve(dataset, range_start, step, range_end)
    charge_indices = [];
    discharge_indices = [];
    impedance_indices = [];
    sampling_steps = range_start:step:range_end;

    %Extract indices of each kind of cycle
    [charge_indices, discharge_indices, impedance_indices] = ExtractCyclesIndices(dataset);
    
    if(range_start < 3.5 || range_end > 4.2)
        error("Range must be between 3.5 and 4.2 volt");
    end

    %Cycle over all the charging cycles, to obtain all the points
    %that more closely corresponds to our steps points. 
  
    %We do so by using the find() function to get indices of the greater or
    %equal value in measured voltage, for each step value of the sampling steps.
    %We use those indices to access the relative time value of each point,
    %and then compute the difference between each subsequent time point.
    %(that's the final feature array)

    features = [];
    targets = [];

    for i=1:length(charge_indices) 
        indices = [];

        %Extract the entire cycle voltages and times, cutting out first 2 values because are "noise".

        %NB: in almost every cycle the first (2-3) voltage measured values are
        %randomly out of the trend of that charging curve (eg. 3.8, 3.2,
        %3.6,3.62,3.64,3.67.....) in particular, the first is way higher than the
        %next ones, and this generates a lot of noise in the features, that results
        %to be 0 in many of these cases.

        full_cycle_voltage = dataset.cycle(charge_indices(i)).data.Voltage_measured(3:end);
        full_cycle_time = dataset.cycle(charge_indices(i)).data.Time(3:end);

        %Extract the indices of the sampling steps value, in the cycle
        for k = 1:length(sampling_steps)
            indices = [indices find(full_cycle_voltage>=sampling_steps(k),1)];
        end
        
        %Extract times at the found indices
        times = full_cycle_time(indices);

        %Compute delta times
        timesteps = diff(times);
        
        features = vertcat(features, timesteps);
        targets = [targets dataset.cycle(discharge_indices(i)).data.Capacity];
    end


    X = features;
    Y = targets;  
  
end