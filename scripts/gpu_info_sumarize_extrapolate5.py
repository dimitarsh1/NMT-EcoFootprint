#!/bin/python

import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
from scipy import interpolate
import numpy as np
import logging

def main():
    """ main function """
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Summarizes the info from the GPU logs.')
    parser.add_argument('-l', '--logfile', required=True, help='The GPU log file.')

    args = parser.parse_args()
    
    # 1. Read and split the files:
    logging.info("Processing ...")
    data = pd.read_csv(os.path.realpath(args.logfile), sep='\t', header=[0])
    data = data.drop(data.index[0])
    
    gpu_power = {}
    gpu_count = 0
    can_collect = True
    
    while(can_collect or gpu_count < 4):
        if data[['gpu', 'pwr']].loc[data['gpu'].to_numpy(int) == gpu_count].to_numpy(float).size != 0:
            gpu_power[gpu_count] = data[['gpu', 'pwr']].loc[data['gpu'].to_numpy(int) == gpu_count].to_numpy(float)
        else:
            can_collect = False
        gpu_count += 1 

    metrics = {}
    for i in gpu_power:
        metrics[i] = []
        # metrics[i] = ["GPU" + str(i)]
        metrics[i].append(str(gpu_power[i].mean(axis = 0)[1]))
        metrics[i].append(str(gpu_power[i].max(axis = 0)[1]))
        # metrics[i].append(str(gpu_power[i].sum(axis = 0)[1]))
        x = np.arange(0, gpu_power[i].shape[0] * 5) # to get ranges from 0 till the end of step 5 (seconds)
        interpolated_values = pd.Series([gpu_power[i][:, 1][v//5] if v%5 == 0 else np.nan for v in x]).interpolate()
        metrics[i].append(str(interpolated_values.sum(axis = 0))) # total
        metrics[i].append(str(interpolated_values.values.sum(axis = 0)/3600000)) # to convert to kWh
    print(",".join([",".join([str(val) for val in arr]) for arr in np.array(list(metrics.values())).T]))
    
    #    print(gpu0.min(axis = 0))
    #    print(gpu0.max(axis = 0))
    #print(data[['gpu', 'pwr', 'gtemp', 'mem']].loc[data['gpu'] == '1'])
    #print(data[['gpu', 'pwr', 'gtemp', 'mem']].loc[data['gpu'] == '2'])
    #print(data[['gpu', 'pwr', 'gtemp', 'mem']].loc[data['gpu'] == '3'])
    
    
if __name__ == "__main__":
    main()
