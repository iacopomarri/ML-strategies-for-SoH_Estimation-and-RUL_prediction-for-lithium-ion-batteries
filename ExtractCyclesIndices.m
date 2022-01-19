function [ch, dis, imp] = ExtractCyclesIndices(dataset)
    charge_indices = [];
    discharge_indices = [];
    impedance_indices = [];

    %Compute 3 arrays containing the indices of the charging,
    %discharging and impedence cycles into the provided dataset.
    
    %Do this way until we find a way to convert struct fields into array, so to
    %apply logical operators on the whole array to get the indices.
    
    %NB: there are some consequent charging phases in B0005 (eg. 23-24)

    for i=1:length(dataset.cycle)
        cycle_type = dataset.cycle(i).type;
     
        if strcmp(cycle_type, 'charge')
            charge_indices = [charge_indices i];

        elseif strcmp(cycle_type, 'discharge')
            discharge_indices = [discharge_indices i];

        else
            impedance_indices = [impedance_indices i] ;      
        end       
    end

    %Return the indices vectors. Ignore first and last charging cycles (they're noise)
    ch = charge_indices(2:end-1); 
    dis = discharge_indices(2:end);
    imp = impedance_indices;

    %% Here we find subsequent charge or discharge cycles. Subsequent means that there are 2 charging cycles without any discharge
    %% Cycle in the middle (and viceversa). Impedance cycles are allowed to be in the middle and subsequent, so they are ignored.
   
    
    %Fill ch and dis with 0 up to the same length
    if length(ch) > length(dis)
    dis(end+1:length(ch)) = 0;
    else ch(end+1:length(dis)) = 0;
    end
    
    %Cycle to check that:
    %charge cycl. i is greater then disch. i-1. If not, charg. i is subsequent and deleted.
    %charge cycl. i is less then disc. i. If not, disch i is subsequent and deleted.
    for i=2:length(ch)
        while ch(i) < dis(i-1) & ch(i)~= 0
            ch(i) = [];
            ch(end+1) = 0;
        end
        while ch(i) > dis(i) & ch(i)~= 0           
            dis(i) = [];
            dis(end+1) = 0;
        end
    end


    ch(find(ch == 0)) = [];
    dis(find(dis == 0)) = [];
    
end