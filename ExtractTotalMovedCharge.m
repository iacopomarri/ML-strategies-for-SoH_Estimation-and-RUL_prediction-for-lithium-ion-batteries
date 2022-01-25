function [Y] = ExtractTotalMovedCharge(dataset)
    
    charge_indices = [];
    discharge_indices = [];
    movedc = [];
    total_movedc = [];

    %Extract indices of each kind of cycle
    [charge_indices, discharge_indices] = ExtractCyclesIndices(dataset);
    
    for i=1:length(charge_indices)
        ch_cycle = dataset.cycle(charge_indices(i)).data;
        dis_cycle = dataset.cycle(discharge_indices(i)).data;
    
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
    
    Y = transpose(total_movedc);
end
