#!/usr/bin/python3
import argparse

# Prepare arguements
parser = argparse.ArgumentParser(description="Check if target hosts are valid.")
parser.add_argument("--f", dest="filtered_websites", default="../collect_quic_traces/filtered_target_websites.txt",
                    help="Filtered websites with QUIC traces file path")
parser.add_argument("--h", dest="target_websites", default="./target_websites.txt",
                    help="Selected websites file path")
parser.add_argument("--o", dest="open_world_websites", default="../collect_quic_traces/open_world_websites.txt",
                    help="Open world websites file path")
parser.add_argument("--n", dest="websites_count", default=10000, type=int,
                    help="Number of websites to be selected")
args = parser.parse_args()

open_world_websites = set()

# Add filtered websites into open world websites first
with open(args.filtered_websites, mode ='r') as filtered_websites:
    for info in filtered_websites:
        url = info.split(";")[0]
        open_world_websites.add(url)

# Add more websites into open world websites from target websites
with open(args.target_websites, mode ='r') as target_websites:
    for target_website in target_websites:
        url = target_website.rstrip()
        if url not in open_world_websites:
            open_world_websites.add(url)
            if len(open_world_websites) >= args.websites_count:
                break

with open(args.open_world_websites, mode ='a') as file_w:
    for url in open_world_websites:
        file_w.write(url + '\n')
