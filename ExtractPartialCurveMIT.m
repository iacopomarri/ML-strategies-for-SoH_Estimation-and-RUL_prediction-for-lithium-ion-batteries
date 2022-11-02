%This function is built for TOYOTA/MIT Li-Ion dataset


%INPUTs: dataset, range of values for the partial curve (mV), step length of the sampling procedure.
%OUTPUTs: X: matrix of features arrays. 
%         Y: Target capacity values. 



function [X,Y] = ExtractPartialCurveMIT(batt, range_start, step, range_end)
    numObservation = size(batt,2);
    
    %% Plot full voltage curves
    
    clc
    sizes = [];
    c = 10;

    figure()
    hold on
    title ('Charging curve. Cycle N. ' + string(c), 'FontSize', 10);
    xlabel('Time' );
    ylabel('Voltage' );

   
    
    for i=1:numObservation
        sizes = [sizes size(batt(i).cycles,2)];
        if size(batt(i).cycles,2) > c
            time = batt(i).cycles(c).t;
            volt = batt(i).cycles(c).V;
            plot(time, volt);
            
        end
    end

    legend('Length: ' + string(sizes), 'location', 'best','FontSize',10);
    %% Cut voltage curves at time 1.54 for irregularities. Plot for cycle=1000
    cycle = 1000;
    timeCut = 1.54;
    
    figure()
    hold on
    legend("Location","southeast");
    legend show
    
    for i=1:numObservation
        for c=1:size(batt(i).cycles,2)
        
            time = batt(i).cycles(c).t;
            volt = batt(i).cycles(c).V;
            idx = find(time > timeCut, 1);
    
            batt(i).cycles(c).t = time(1:idx);
            batt(i).cycles(c).V = volt(1:idx);
        
            if c == cycle
                plot(time(1:idx), volt(1:idx));
            end
        end
    end
    
    
    %% Computing the features for each cycle for each battery
    
    sampling_steps = range_start:step:range_end;
    %voltage_points = range_start:0.01:range_end;
    
    
    X = cell(numObservation, 1);
    Y = cell(numObservation,1);
    
    for i=1:numObservation
        features = [];
    
        for c=1:size(batt(i).cycles,2)
        
            time = batt(i).cycles(c).t;
            volt = batt(i).cycles(c).V;
            
            %Make voltage array values unique, needed for interpolation
            [volt,ia] = unique(volt, "stable"); 
            time = time(ia);

            disp('B : ' + string(i) + ', C : ' + string(c));
            interpolation = fit(volt, time, 'linear');
            time_points = interpolation(sampling_steps);
            time_steps = diff(time_points);
            features = vertcat(features, time_steps');
    
        end
        X{i} = features;
        Y{i} = batt(i).summary.QDischarge;
    end
end
