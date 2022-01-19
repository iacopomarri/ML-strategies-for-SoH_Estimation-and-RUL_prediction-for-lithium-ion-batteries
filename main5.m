%% Structure Emil's data into a NASA like data structure
clc
close all
clear all

B8 = load('I_8_20.mat');
B25 = load('I_25_02.mat');
B50 = load('I_50.mat');


%% Manually clean dirty cycles

batteries = {B8, B25, B50};
batteries{2}.istanti_finali(7) = [];
batteries{2}.istanti_iniziali(7) = [];
batteries{2}.istanti_finali(1) = [];
batteries{2}.istanti_iniziali(1) = [];


%% Find max voltage charging value, use it to cut che curve after the discharge is over --> obtain the charge curve

close all
cutoff_charge = [];

for j=1:length(batteries)
    batt = batteries{j};
    maxs = [];
    figure()
    legend show
    hold on

    for i=1:1:length(batt.istanti_finali)
        start = batt.istanti_finali(i);
        plot(batt.t(start:start+5000) -batt.t(start), batt.V(start:start+5000), "DisplayName", "Cycle " +int2str(i))
        
        
        maxs = [maxs max(batt.V(start:start+5000))];
    end

    cutoff_charge = [cutoff_charge min(maxs)];
end



%% Build struct
clc
close all
cycles = {};

for j=1:length(batteries)
    batt = batteries{j};
    mystruct = [];
    charge_indeces = 3:2:length(batt.istanti_finali)*2+1;
    discharge_indeces = 2:2:length(batt.istanti_finali)*2;

    figure()
    legend show
    hold on
    
    for i=1:length(discharge_indeces)
        start_disch = batt.istanti_iniziali(i);
        start_charge = batt.istanti_finali(i);
        end_charge = find(batt.V(start_charge:start_charge+5000)>=cutoff_charge(j) ,1) + start_charge;
    
        plot(batt.t(start_disch: end_charge) -batt.t(start_disch), batt.V(start_disch: end_charge));
        hold on
    
        mystruct(discharge_indeces(i)).type='discharge';
        mystruct(discharge_indeces(i)).data.Voltage_measured = transpose(batt.V(start_disch: start_charge));
        mystruct(discharge_indeces(i)).data.Current_measured = transpose(batt.I(start_disch: start_charge));
        mystruct(discharge_indeces(i)).data.Time = transpose(batt.t(start_disch: start_charge)-batt.t(start_disch));
        mystruct(discharge_indeces(i)).data.Temperature_measured = transpose(batt.Tem(start_disch: start_charge));
        mystruct(discharge_indeces(i)).data.Capacity = batt.C(i);
    
        mystruct(charge_indeces(i)).type='charge';
        mystruct(charge_indeces(i)).data.Voltage_measured = transpose(batt.V(start_charge: end_charge));
        mystruct(charge_indeces(i)).data.Current_measured = transpose(batt.I(start_charge: end_charge));
        mystruct(charge_indeces(i)).data.Time = transpose(batt.t(start_charge: end_charge)-batt.t(start_charge));
        mystruct(charge_indeces(i)).data.Temperature_measured = transpose(batt.Tem(start_charge: end_charge));
    end

    %%Manually insert first charge
    mystruct(1).type='charge';
    mystruct(1).data.Voltage_measured = transpose(batt.V(1:batt.istanti_iniziali(1)));
    mystruct(1).data.Current_measured = transpose(batt.I(1:batt.istanti_iniziali(1)));
    mystruct(1).data.Time = transpose(batt.t(1:batt.istanti_iniziali(1))-batt.t(1));
    mystruct(1).data.Temperature_measured = transpose(batt.Tem(1:batt.istanti_iniziali(1)));

    cycles{j} = mystruct;
end

%% Plotting the voltage values into the time windows provided by "istanti_iniziali" e "istanti_finali"
load("I_8_20.mat")
close all
figure()
xlabel("Time (rescaled to 0 for each cycle)");
ylabel("Voltage");
title ('Discharge + Charge');
legend show;
hold on
for i=1:5:80
    a = istanti_iniziali(i);
    b = istanti_finali(i);

    plot(t(a-500:b+5000) -t(a-500), V(a-500:b+5000));
  
    pause(0.2);
end

%% Plotting the first cycles of all 3 batteries
for j=1:length(batteries)
    batt = batteries{j};
    plot(batt.t(end-30000:end) - batt.t(end-30000), batt.V(end-30000:end), "DisplayName", "Battery " +int2str(j));
    legend show
    hold on
end
