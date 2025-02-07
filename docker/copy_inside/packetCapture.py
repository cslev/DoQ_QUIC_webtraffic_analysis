#!/usr/bin/python3
import argparse
import multiprocessing
import nest_asyncio
import os
import pyshark
import subprocess
import time
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Resolve async event loop
nest_asyncio.apply()

# Prepare arguements
parser = argparse.ArgumentParser(description="Capture HTTP3 traffic.")
parser.add_argument("--trace_file_dir", dest="trace_file_dir", default="./traces/",
                    help="trace files directory")
parser.add_argument("--target_websites", dest="target_websites", default="./target_websites.txt",
                    help="target hosts file path")
parser.add_argument("--target_websites_status", dest="target_websites_status", default="./target_websites_status.txt",
                    help="target websites status after filter")
parser.add_argument("--websites_count", dest="websites_count", default=10, type=int,
                    help="total websites to keep")
parser.add_argument("--gap_count", dest="gap_count", default=0, type=int,
                    help="mark the start of target websites capture")
parser.add_argument("--access_count", dest="access_count", default=2, type=int,
                    help="numbers of access for each host")
parser.add_argument('--filter', action='store_true', default=False,
                    help='enable filter over target websites with QUIC traces')
args = parser.parse_args()

# Google Chromium web driver
def open_website(url, sslkeylog_file):
    os.environ['SSLKEYLOGFILE'] = sslkeylog_file
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument(f"--user-data-dir=./google_chrome_data/")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    print(driver.title)

# tshark trace capture
def trace_capture(shell_command):
    print("-----Running tshark")
    subprocess.run(shell_command, shell=True)

# AdGuard DoQ proxy
def doq_proxy(shell_command):
    print("-----Running Doq")
    subprocess.run(shell_command, shell=True)

# url: website address, index: trace file index, output path: trace file path
def packet_capture(url, index, output_path):
    # Save system dns configuration
    subprocess.run("cp /etc/resolv.conf .", shell=True)
    # Replace system dns configuration with DoQ proxy
    subprocess.run("cp ./dns_config/resolv.conf /etc/resolv.conf", shell=True)
    trace_file, trace_file_dir_path, website_dir_path = "", "", ""
    try:
        domain = url.split("//")[1]
        dir_name = "_".join(domain.split("."))
        dir_name = "_".join(dir_name.split("/"))[:-1]
        website_dir_path = f"{output_path}{dir_name}/"
        # Create a subdirectory for each website
        if not os.path.isdir(website_dir_path):
            os.makedirs(website_dir_path)
        # Create a subdirectory for each visit trace
        trace_file_dir_path = f"{website_dir_path}{index+ 1}_{dir_name}/"
        # Remove previous one
        if os.path.isdir(trace_file_dir_path):
            subprocess.run(f"rm -rf {trace_file_dir_path}", shell=True)
        os.makedirs(trace_file_dir_path)
        trace_file = f"{trace_file_dir_path}{index+ 1}_{dir_name}.pcap"
        sslkeylog_file = f"{trace_file_dir_path}{index+ 1}_{dir_name}.log"
        subprocess.run(["touch", f"{trace_file}"])
        subprocess.run(["chmod", "o=rw", f"{trace_file}"])
        subprocess.run(["touch", f"{sslkeylog_file}"])
        # Process1: run tshark to capture network packets
        #process_tshark = multiprocessing.Process(target=trace_capture, args=(f"tshark -i {target_interface} -w {trace_file} -f 'port 80 or port 443 or port 853'",))
        process_tshark = multiprocessing.Process(target=trace_capture, args=(f"tcpdump -w {trace_file} -f 'port 80 or 443 or 853'",))
        # Process2: visit target website
        process_web_driver = multiprocessing.Process(target=open_website, args=(url, sslkeylog_file,))
        # Process3: start DoQ proxy
        process_doq_proxy = multiprocessing.Process(target=doq_proxy, args=("/root/go/bin/dnsproxy -u quic://94.140.14.14/ -l 127.0.0.1",))
        process_tshark.start()
        # Sleep for 1 second to ensure tshark is up
        time.sleep(1)
        process_doq_proxy.start()
        process_web_driver.start()
        process_web_driver.join()
        # process_doq_proxy.terminate()
        # process_tshark.terminate()
        subprocess.run("pkill tcpdump", shell=True)
        subprocess.run("python3 kill_doq.py", shell=True)
        time.sleep(1)
    except:
        subprocess.run(f"rm -rf {trace_file_dir_path}", shell=True)
    finally:
        # Restore system dns configuration
        subprocess.run("cp ./resolv.conf /etc/", shell=True)
    return trace_file, trace_file_dir_path, website_dir_path

def contains_quic(trace_file):
    try:
        trace_file = pyshark.FileCapture(trace_file, display_filter="quic")
        for packet in trace_file:
            for layer in packet:
                if layer.layer_name == 'quic':
                    if packet.udp.srcport == "443" or packet.udp.dstport == "443":
                        return True
        return False
    except:
        return False

def parse_pcap(trace_file_path):
    try:
        host_ip = os.popen("hostname -i").read().rstrip("\n")
        trace_file = pyshark.FileCapture(trace_file_path)
        features = list()
        for packet in trace_file:
            feature = dict()
            # Parse protocol: QUIC 1, TCP 0
            if packet.transport_layer == "QUIC" or packet.transport_layer == "UDP":
                feature["protocol"] = "1"
            else:
                feature["protocol"] = "0"
            # Parse packet size
            feature["length"] = str(packet.length)
            # Parse relative time
            feature["relative_time"] = str("{:.9f}".format(float(getattr(packet.frame_info, "time_delta"))))
            # Parse direction: out 1, in 0
            feature["direction"] = "1" if str(packet.ip.src_host) == host_ip else "0"
            # Parse source ip
            feature["src_ip"] = str(packet.ip.src_host)
            # Parse source port
            feature["src_port"] = str(packet[packet.transport_layer].srcport)
            # Parse destination ip
            feature["dst_ip"] = str(packet.ip.dst_host)
            # Parse destination port
            feature["dst_port"] = str(packet[packet.transport_layer].dstport)
            features.append(feature)
        return features
    except:
        return []

def generate_quic_traces(target_websites_dir):
    # Keep websites that contain QUIC traces with hit rate
    hits_status = defaultdict(list)
    for i in range(0, args.access_count):
        with open(target_websites_dir) as target_websites:
            websites_count = 0
            gap_count = 0
            for url in target_websites:
                # Adjust starting point of target websites list
                gap_count += 1
                if gap_count <= args.gap_count:
                    continue
                url = url.rstrip("\n")
                # Capture pcap and ssh key log files
                trace_file_path, trace_file_dir_path, website_dir_path = packet_capture(url, i, args.trace_file_dir)
                # Check if it contains quic traces
                has_quic = False
                if contains_quic(trace_file_path):
                    hits_status[url].append(i + 1)
                    has_quic = True
                # Parse pcap file
                features = parse_pcap(trace_file_path)
                # Remove the pcap file
                subprocess.run(f"rm -rf {trace_file_dir_path}", shell=True)
                dataset_file_path = website_dir_path + trace_file_path.split("/")[-1].split(".")[0] + ".csv"
                if len(features) != 0:
                    if not args.filter or has_quic:
                        print(dataset_file_path)
                        with open(dataset_file_path, mode ='w') as dataset_file:
                            dataset_file.write(";".join(features[0].keys()) + '\n')
                            for feature in features:
                                dataset_file.write(";".join(feature.values()) + '\n')
                # Record hit rate
                if i == args.access_count - 1:
                    hits = hits_status[url]
                    hit_rate = 0.0
                    if len(hits) != 0:
                        hit_rate = len(hits) / args.access_count
                    with open(args.target_websites_status, mode ='a') as file_w:
                        file_w.write(f"{url};{hit_rate};{hits}\n")
                # Check number of websites limit
                websites_count += 1
                if websites_count == args.websites_count:
                    break

if __name__ == '__main__':
    start_time = time.time()
    # Get the first avaiable network interface
    target_interface = str()
    interfaces = subprocess.run(["tshark", "-D"], stdout=subprocess.PIPE).stdout.decode("utf-8")
    target_interface = interfaces.split("\n")[0].split(" ")[1].rstrip("\n")

    # Get host ip
    host_ip = os.popen("hostname -i").read().rstrip("\n")
    reserved_ips = {host_ip, "94.140.14.14"}

    # Generate QUIC traces on filtered target websites
    generate_quic_traces(args.target_websites)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
