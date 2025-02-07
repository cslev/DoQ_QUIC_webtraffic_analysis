#!/usr/bin/python3
import argparse
import pandas as pd

# Prepare arguements
parser = argparse.ArgumentParser(description="Check if target hosts are valid.")
parser.add_argument("--f", dest="filtered_websites", default="./filtered_target_websites.txt",
                    help="Filtered websites with QUIC traces file path")
parser.add_argument("--t", dest="target_websites_status", default="./target_websites_status.txt",
                    help="Selected websites status file path")
parser.add_argument("--n", dest="websites_count", default=500, type=int,
                    help="Number of websites to be selected")
parser.add_argument("--h", dest="hit_rate", default=0.8, type=float,
                    help="Threshold to keep as closed world websites")
args = parser.parse_args()

df = pd.read_csv(args.target_websites_status, sep=';', header=None)

filtered_websites_list = list()

for index, row in df.iterrows():
    url = row[0]
    hit_rate = float(row[1])
    if len(filtered_websites_list) >= args.websites_count:
        break
    if hit_rate >= args.hit_rate:
        filtered_websites_list.append((url, hit_rate))

with open(args.filtered_websites, mode ='a') as file_w:
    for (url, hit_rate) in filtered_websites_list:
        # file_w.write(url + '\n')
        file_w.write(url + ';' + str(hit_rate) + '\n')
