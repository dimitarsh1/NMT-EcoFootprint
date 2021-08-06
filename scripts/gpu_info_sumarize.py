#!/bin/python

import datetime
import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
from scipy import interpolate
import numpy as np
import logging

def elapsed_time(data):
    ''' Computes the elapsed time for the current data (specific format is needed)
    
    :param data: Dataframe
    :returns: elapsed time in minutes
    '''
    start = data[["#Date", "Time"]].to_numpy()[0]
    end = data[["#Date", "Time"]].to_numpy()[-1]

    start_date = datetime.datetime.strptime(" ".join([str(x) for x in start]), "%Y%m%d %H:%M:%S")
    end_date = datetime.datetime.strptime(" ".join([str(x) for x in end]), "%Y%m%d %H:%M:%S")
    start_timestamp = datetime.datetime.timestamp(start_date)
    end_timestamp = datetime.datetime.timestamp(end_date)

    return (end_timestamp - start_timestamp), (end_timestamp - start_timestamp) / 60, (end_timestamp - start_timestamp) / 3600 # to get seconds, minutes and then hours
    
    
def main():
    """ main function """
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Summarizes the info from the GPU logs.')
    parser.add_argument('-l', '--logfile', required=True, help='The GPU log file.')
    parser.add_argument('-r', '--region', required=False, default='NL', help='The region where the experiments ran.')
    parser.add_argument('-e', '--experiment', required=False, default='', help='Which experiment to print.')

    args = parser.parse_args()
    
    #CONSTANTS: (source: https://github.com/Breakend/experiment-impact-tracker)
    #carbon_intensity = {'NL': 462.73049119238584, 'IE': 395.81760304466565}
    carbon_intensity = {'NL': {'mean': 399.368549233678, 'std': 31.9251119491906}, 'IE': {'mean': 229.871843212371 , 'std': 77.4025581328516}} #average for first half of 2020
    PUE = 1.59 # Power Usage Effectiveness for 2020
    
    experiment_name = {
            'gpu.log.tab.clean': "Train",
            'gpu_trans.log.tab.clean': "Translate",
            'gpu_trans_ct2.log.tab.clean': "Translate (Quantized)",
            'gpu_trans_ct2_int8.log.tab.clean': "Translate (Quantized int8)",
            'gpu_trans_ct2_int16.log.tab.clean': "Translate (Quantized int16)"
        }

    labels = {
            # "exp1": ["Min", "Max", "Average", "Total (kWh)", "Total Effective (kWh * PUE)"],
            "exp0": ["Name", "GPU time (h)"],
            "exp1": ["Average", "Total (kWh)", "Total Effective (kWh * PUE)"],
            "exp2": ["Name", "Elapsed time (h)", "Avg. power draw", "kWh", "CO2 (kg)"],
            "exp3": ["Name", "Frame buffer (MB)", "SM (\%)", "Mem (\%)"],
            "exp4": ["Name", "Temp (C)"],
            "exp5": ["Name", "Frame buffer (MB)", "SM (\%)", "Mem (\%)"]
        }
        
    # 0. Let's print a table with the data for all GPUs + CO2 estimate
    #expname = '-'.join([x for x in os.path.split(os.path.split(args.logfile)[0])[-1].split('-')[:2]]) + " " + \
    #         os.path.split(os.path.split(args.logfile)[0])[-1].split('-')[2] #+ " " + experiment_name[os.path.split(args.logfile)[-1]]
    expname = "NeutralRewriter"

    # 1. Read and split the files:
    logging.info("Processing ...")
    data = pd.read_csv(os.path.realpath(args.logfile), sep='\t', header=[0])
    data = data.drop(data.index[0])
    
    gpu_power = {}
    gpu_memory = {}
    gpu_count = 0
    can_collect = True
    
    gpu_power_all = {}
    while(gpu_count < 4):
        if data[['gpu']].loc[data['gpu'].to_numpy(int) == gpu_count].to_numpy(float).size != 0:
            gpu_power[gpu_count] = data[['gpu', 'pwr']].loc[data['gpu'].to_numpy(int) == gpu_count].to_numpy(float)
            gpu_power_all[gpu_count] = data['pwr'].loc[data['gpu'].to_numpy(int) == gpu_count].to_numpy(float)
            gpu_memory[gpu_count] = data[['gpu', 'fb', 'gtemp', 'sm', 'mem']].loc[data['gpu'].to_numpy(int) == gpu_count].to_numpy(float)
        else:
            pass
        gpu_count += 1 

    gputime = elapsed_time(data) # get elapsed time
    
    power = {}
    power_total = []
    for i in gpu_power:
        power[i] = []
        power[i].append("GPU" + str(i))
        power[i].append(str(gpu_power[i].min(axis = 0)[1].round(2))) # Get the min
        power[i].append(str(gpu_power[i].max(axis = 0)[1].round(2))) # Get the max
        power[i].append(str(gpu_power[i].mean(axis = 0)[1].round(2))) # Get the mean / average
        gpu_hours = gputime[2]
        kwh = (gpu_power[i].sum(axis = 0)[1]/3600000).round(2)
        power[i].append(str(kwh)) # GPU kw Hours
        power[i].append(str(round(PUE * kwh, 2))) # Effective GPU Hours
        power[i].append(str(PUE * carbon_intensity[args.region]['mean'] * kwh)) # to comput CO2
        power[i].append(str(PUE * carbon_intensity[args.region]['std'] * kwh)) # to comput CO2
    
    #power_total.append(round(gputime[0], 2)) # seconds
    #power_total.append(round(gputime[1], 2)) # minutes
    power_total.append(round(gputime[2], 2)) # hours
    power_total.append((sum([gpu_power[i].sum(axis = 0)[1] for i in gpu_power])/sum([len(gpu_power[i]) for i in gpu_power])).round(2)) # mean
    #power_total.append(np.concatenate([gpu_power_all[i] for i in gpu_power_all]).min())
    #power_total.append(np.concatenate([gpu_power_all[i] for i in gpu_power_all]).max())
    #power_total.append(np.concatenate([gpu_power_all[i] for i in gpu_power_all]).std())
    #power_total.append(np.concatenate([gpu_power_all[i] for i in gpu_power_all]).mean())
    total_Watts = sum([gpu_power[i].sum(axis = 0)[1] for i in gpu_power])
    total_len = sum([len(gpu_power[i]) for i in gpu_power])
    #power_total.append(sum([gpu_power[i].mean(axis = 0)[1] for i in gpu_power]).round(2))
    
    print(str(total_Watts) + " " + str(total_len))
        
    #power_total.append((sum([gpu_power[i].sum(axis = 0)[1] for i in gpu_power])/sum([len(gpu_power[i]) for i in gpu_power])).round(2)) 
    total_power = (sum([gpu_power[i].sum(axis = 0)[1] for i in gpu_power])/3600000).round(2) # to convert to kWh
    power_total.append(total_power)
    power_total.append(str(round(PUE * carbon_intensity[args.region]['mean'] * total_power / 1000, 2))) # to compute CO2
    power_total.append(str(round(PUE * carbon_intensity[args.region]['std'] * total_power / 1000, 2))) # to compute the deviation CO2
    
    memory = {}
    memory_total = []
    for i in gpu_memory:
        memory[i] = []
        memory[i].append("GPU" + str(i))
        memory[i].append(str(gpu_memory[i].mean(axis = 0)[1].round(2))) # Get the mean / average framebuffer
        memory[i].append(str(gpu_memory[i].mean(axis = 0)[3].round(2))) # Get the mean / average SM
        memory[i].append(str(gpu_memory[i].mean(axis = 0)[4].round(2))) # Get the mean / average mem
        memory[i].append(str(gpu_memory[i].mean(axis = 0)[2].round(2))) # Get the temperature
    
    for ii in [1, 3, 4, 2]: # 1 = fb, 3 = SM, 4 = mem, 2 = temp
        memory_total.append((sum([gpu_memory[i].sum(axis = 0)[ii] for i in gpu_memory])/sum([len(gpu_memory[i]) for i in gpu_memory])).round(2)) # average fb over all gpus
    
    if (args.experiment == '0' or args.experiment == ''): #run time
        print("-----------------------------------")
        # EXPERIMENT 0 to table: 
        print(" & " + " & ".join([label for label in labels['exp0']]) + "\\\\\hline")
        print(expname + " & " + str(round(gputime[2], 2)) + "\\\\\hline") # it's just the average temp
        
    if (args.experiment == '1' or args.experiment == ''):
        print("-----------------------------------")
        # EXPERIMENT 1 to table: 
        print(" & " + " & ".join(["\multicolumn{4}{c}{" + power[i][0] + "}" for i in power]) + "\\\\\hline")
        print(" & " + " & ".join([label for label in labels['exp1']]) + "\\\\\hline")
        print(expname + " & " + " & ".join([" & ".join([str(power[i][x+1]) for x in range(len(power[i])-2)]) for i in power]) + "\\\\\hline")

    if (args.experiment == '2' or args.experiment == ''):
        print("-----------------------------------")
        # EXPERIMENT 2 to table: 
        print(" & " + " & ".join([label for label in labels['exp2']]) + "\\\\\hline")
        print(expname + " & " + " & ".join([str(pt) for pt in power_total]) + "\\\\\hline")

    if (args.experiment == '3' or args.experiment == ''):
        print("-----------------------------------")        
        # EXPERIMENT 3 to table: 
        print(" & " + " & ".join(["\multicolumn{4}{c}{" + power[i][0] + "}" for i in power]) + "\\\\\hline")
        print(" & " + " & ".join([label for label in labels['exp3']]) + "\\\\\hline")
        print(expname + " & " + " & ".join([" & ".join([str(memory[i][x+1]) for x in range(len(memory[i])-2)]) for i in memory]) + "\\\\\hline")

    if (args.experiment == '4' or args.experiment == ''):
        print("-----------------------------------")    
        # EXPERIMENT 4 to table: 
        print(" & " + " & ".join([label for label in labels['exp4']]) + "\\\\\hline")
        print(expname + " & " + str(memory_total[-1]) + "\\\\\hline") # it's just the average temp
    
    if (args.experiment == '5' or args.experiment == ''):
        # print("-----------------------------------")    
        # EXPERIMENT 5 to table: 
        print("\t" + "\t".join([label for label in labels['exp5']]))
        print(expname + "\t" + "\t".join([str(mt) for mt in memory_total]))
        
        
    #print("-----------------------------------")    
    logging.info("Finished.")
#    print(" & ".join([str(x) for x in power[i]]))
    
#    print(",".join([",".join([str(val) for val in arr]) for arr in np.array(list(power.values())).T]))
    
    #    print(gpu0.min(axis = 0))
    #    print(gpu0.max(axis = 0))
    #print(data[['gpu', 'pwr', 'gtemp', 'mem']].loc[data['gpu'] == '1'])
    #print(data[['gpu', 'pwr', 'gtemp', 'mem']].loc[data['gpu'] == '2'])
    #print(data[['gpu', 'pwr', 'gtemp', 'mem']].loc[data['gpu'] == '3'])
    # we want:
    # Experiment name & Total power (kwh) & GPU hours & Estimated CO2 impact
    # Experiment name (train) average power (w) & GPU 1 & GPU 2 & GPU 3 & GPU 4
    # Experiment name (test) average power (w) & GPU 1 
    # Experiment name (train) & average memory consumption
    # Experiment name (train) & average memory consumption
    
if __name__ == "__main__":
    main()
