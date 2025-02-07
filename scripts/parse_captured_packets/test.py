#!/usr/bin/python3
import argparse
import nest_asyncio
import pyshark
import os

# Resolve async event loop
nest_asyncio.apply()

# Prepare arguements
parser = argparse.ArgumentParser(description="Parse captured packets.")
parser.add_argument("--o", dest="trace_file_dir", default="../collect_quic_traces/traces/",
                    help="trace files directory")
parser.add_argument("--d", dest="dataset_dir", default="./dataset/",
                    help="dateset directory")
parser.add_argument("--p", dest="protocol", default="quic",
                    help="protocol to be parsed")
args = parser.parse_args()

# Create a new dataset directory if not exist
if not os.path.isdir(args.dataset_dir):
    os.makedirs(args.dataset_dir)

def parse_trace_file(file_path):
    trace_file_info = dict()
    trace_file = pyshark.FileCapture(file_path)
    for packet in trace_file:
        packet_number = int(packet.frame_info.number)
        trace_file_info[packet_number] = dict()
        attributes = packet.frame_info.field_names
        # print(attributes)
        for attr in attributes:
            key = "#FRAME#" + str(attr)
            value = getattr(packet.frame_info, attr)
            trace_file_info[packet_number][key] = value
        for layer in packet:
            # Discard Ethernet and data layer
            if str(layer.layer_name).upper() in ["ETH", "DATA"]:
                continue
            # print(f"__________{layer.layer_name}__________")
            for attr in layer.field_names:
                # Discard payload
                if str(attr) == "payload":
                    continue
                layer_name = str(layer.layer_name).upper()
                key = f"#{layer_name}#" + str(attr)
                value = getattr(layer, attr)
                trace_file_info[packet_number][key] = value
    return trace_file_info

def test_single_trace(file_path):
    # domain_ips = get_domain_ips(file_path)
    # domain_ips.discard(host_ip)
    # domain_ips.discard("94.140.14.14")
    # print(domain_ips)
    # frame_nums = get_frame_nums(file_path, domain_ips)
    # print(frame_nums)
    # features = parse_selected_frame(file_path, frame_nums)
    # print(features)
    trace_file_info = parse_trace_file(file_path)
    keys = set()
    for dic in trace_file_info.values():
        keys.update(dic.keys())
    keys = sorted(keys)
    print(";".join(keys))
    for frame_num, frame_info in trace_file_info.items():
        res = ""
        for key in keys:
            if key in frame_info:
                res += frame_info[key] + ";"
            else:
                res += "NA;"
        print(res)
    # print(keys)

# test_single_trace("/users/KevinHZ/NUS_Capstone/scripts/collect_quic_traces/traces/www_google_com/1_www_google_com/1_www_google_com.pcap")

def parse_traces():
    for domain in os.listdir(args.trace_file_dir):
        cur_domain = domain
        print(cur_domain)
        cur_domain_dir = f"{args.dataset_dir}{cur_domain}/"
        # Create a subdirectory for each domain
        if not os.path.isdir(cur_domain_dir):
            os.makedirs(cur_domain_dir)
        domain_path = f"{args.trace_file_dir}{cur_domain}/"
        # print(domain_path)
        for domain_sub_path in os.listdir(domain_path):
            trace_file_path = f"{domain_path}{domain_sub_path}/"
            for trace_file in os.listdir(trace_file_path):
                if ".pcap" in trace_file:
                    trace_file_info = parse_trace_file(trace_file_path + trace_file)
                    keys = set()
                    for dic in trace_file_info.values():
                        keys.update(dic.keys())
                    keys = sorted(keys)
                    dataset_file_path = cur_domain_dir + trace_file.split("_")[0] + ".csv"
                    print(dataset_file_path)
                    with open(dataset_file_path, mode ='w') as dataset_file:
                        dataset_file.write(";".join(keys) + ';\n')
                        for frame_num, frame_info in trace_file_info.items():
                            res = ""
                            for key in keys:
                                if key in frame_info:
                                    res += frame_info[key] + ";"
                                else:
                                    res += "NA;"
                            dataset_file.write(res + '\n')
parse_traces()
