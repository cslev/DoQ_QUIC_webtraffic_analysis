#!/usr/bin/python3
import argparse
import subprocess
import csv
# Prepare arguements
parser = argparse.ArgumentParser(description="Check if target hosts are valid.")
parser.add_argument("--t", dest="target_websites", default="./tranco-top-1m.csv",
                    help="Target websites file path")
parser.add_argument("--h", dest="selected_hosts", default="./target_websites.txt",
                    help="Selected websites file path")
parser.add_argument("--n", dest="websites_count", default=100, type=int,
                    help="Number of websites to be selected")
args = parser.parse_args()

selected_domain = set()
selected_websites = set()
websites_count = int(args.websites_count)
visited_websites_count = 0

with open(args.target_websites, mode ='r') as file_r:
    # reading the CSV file
    target_websites = csv.reader(file_r)
    for website_info in target_websites:
        visited_websites_count += 1
        url = f"https://{website_info[1]}/"
        domain = website_info[1].split(".")[0]
        print(url, domain)
        print(f"====={url} {len(selected_websites)} {visited_websites_count}∆=====")
        # Send HTTP request
        process_curl = subprocess.Popen(f"curl {url} -IL --max-time 10", shell=True, stdout=subprocess.PIPE)
        stdout, stderr = process_curl.communicate()
        try:
            # Select only domains with alternative services containing either the string “h3” (HTTP/3) or “quic”
            has_quic = False
            moved_permantly = False
            valid = False
            for line in str(stdout.decode()).split("\n"):
                if "HTTP" in line and ("301" in line or "308" in line):
                    moved_permantly = True
                if "HTTP" in line and "200" in line:
                    valid = True
                # Find final url via redirect
                if moved_permantly and "ocation:" in line:
                    new_url = line.split(" ")[1]
                    if "http" in new_url:
                        url = new_url
                    else: # Special redirect: /en/, /v3/
                        url = url.strip().rstrip("/") + new_url
                    moved_permantly = False
                if "alt-svc" in line:
                    if "h3" in line or "quic" in line:
                        has_quic = True
                        print(stdout.decode())
            if has_quic and valid:
                # Filter domains with the same private domain name but different public suffixes
                if domain in selected_domain or url in selected_websites:
                    continue
                selected_domain.add(domain)
                selected_websites.add(url)
                websites_count -= 1
                print(f"-----Target added: {url}-----")
                with open(args.selected_hosts, mode ='a') as file_w:
                    file_w.write(url + '\n')
            if websites_count == 0:
                break
        except:
            continue
