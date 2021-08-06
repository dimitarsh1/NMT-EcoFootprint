#!/bin/python

import os
import argparse
import pandas as pd
import logging
import altair as alt
from vega_datasets import data

def main():
    """ main function """
    # read argument - file with data
    parser = argparse.ArgumentParser(description='Summarizes the info from the GPU logs.')
    parser.add_argument('-i', '--input', required=True, help='The CSV file (three colums).')
    
    args = parser.parse_args()
    
    logging.info("Processing ...")
    data = pd.read_csv(os.path.realpath(args.input), sep='\t', header=[0])
    
    fossil = [sum(data[['Fossil']][i*24:(i+1)*24].to_numpy(float))/24 for i in range(len(data)//24)]
    renewable = [sum(data[['Renewable']][i*24:(i+1)*24].to_numpy(float))/24 for i in range(len(data)//24)]
    timestamp = [sum(data[['Timestamp']][i*24:(i+1)*24].to_numpy(float))/24 for i in range(len(data)//24)]
    
    print(fossil)
    print(renewable)
    proc_data = pd.DataFrame({'x': timestamp, 'y1': fossil, 'y2': renewable})
    
    chart = alt.Chart(proc_data).mark_bar().encode(
        x='x',
        y='y1'
    )

    chart.show()

    
if __name__ == "__main__":
    main()
