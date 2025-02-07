#!/usr/bin/python3
import argparse
import pandas as pd
import numpy as np
import os
import csv

parser = argparse.ArgumentParser(description="Produce dataset in numpy array format.")
parser.add_argument("--num_datapoints", dest="num_datapoints", default=4, type=int,
                    help="number of datapoins to keep per class")
parser.add_argument("--dataset_open_dir", dest="dataset_open_dir", default="/mydata/dataset_open/",
                    help="open-world dataset csv files directory")
args = parser.parse_args()

def filtered_open_world(dataset_path, domain_index_mapping_path, num_datapoints: int = 4):
    domain_count = 0
    stats = [0] * num_datapoints
    # Get domain to index mapping
    df = pd.read_csv(domain_index_mapping_path, sep=';', header=0)
    domain_to_index = df.set_index("url")["index"].to_dict()
    for domain in os.listdir(dataset_path):
        if domain in domain_to_index:
            continue
        domain_dir = os.path.join(dataset_path, domain)
        file_count = len([f for f in os.listdir(domain_dir)])
        for i in range(1, num_datapoints + 1):
            if file_count >= i:
                stats[i - 1] += 1
    return stats


if __name__ == '__main__':
    domain_index_mapping_path = "./domain_index_mapping.csv"
    stats = filtered_open_world(args.dataset_open_dir, domain_index_mapping_path, args.num_datapoints)
    for index, value in enumerate(stats):
        print(f'File Count: {index + 1}, Websites Count: {value}')
