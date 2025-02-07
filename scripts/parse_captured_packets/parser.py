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

# Dataset structure: targetHostName: frameNum: fieldName: fieldValue
# dataset = dict()

# Create a new dataset directory if not exist
if not os.path.isdir(args.dataset_dir):
    os.makedirs(args.dataset_dir)

# Get all domain ips within the quic packet
def get_domain_ips(trace_file_path):
    domain_ips = set()
    trace_file = pyshark.FileCapture(trace_file_path, display_filter="quic")
    for packet in trace_file:
        for layer in packet:
            # if layer.layer_name == 'quic':
            domain_ips.add(packet.ip.dst_host)
            domain_ips.add(packet.ip.src_host)
    return domain_ips

# Get all related frame numbers from domain ips
def get_frame_nums(trace_file_path, domain_ips):
    frame_nums = list()
    trace_file = pyshark.FileCapture(trace_file_path)
    for packet in trace_file:
        for layer in packet:
            if layer.layer_name == 'ip':
                if packet.ip.dst_host in domain_ips or packet.ip.src_host in domain_ips:
                    frame_nums.append(int(packet.frame_info.number) - 1)
    return frame_nums

# Extract features from selected frame numbers
def parse_selected_frame(trace_file_path, frame_nums):
    trace_file = pyshark.FileCapture(trace_file_path)
    features = list()
    prev_time = float(trace_file[frame_nums[0]].sniff_timestamp)
    for frame_num in frame_nums:
        feature = dict()
        frame = trace_file[frame_num]
        # print(dir(frame))
        # print(frame_num + 1, frame.transport_layer, frame.length, frame.sniff_timestamp)
        feature["frame_num"] = str(frame_num)
        if frame.transport_layer == "TCP":
            feature["protocol"] = "TCP"
        elif frame.highest_layer == "QUIC":
            feature["protocol"] = "QUIC"
        else:
            # print(f"!!!!!{frame.transport_layer} {frame.highest_layer}")
            continue
        feature["length"] = str(frame.length)
        # print(float(frame.sniff_timestamp), prev_time, float(frame.sniff_timestamp) - prev_time)
        feature["inter_arrival_time"] = str(format(float(frame.sniff_timestamp) - prev_time, ".8f"))
        feature["direction"] = "out" if str(frame.ip.src_host) == host_ip else "in"
        prev_time = float(frame.sniff_timestamp)
        # print(feature)
        features.append(feature)
    return features

# host_ip = os.popen("hostname -i").read().rstrip("\n")
# print(host_ip)

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
            # if str(layer.layer_name).upper() in ["ETH", "DATA"]:
            #     continue
            # print(f"__________{layer.layer_name}__________")
            for attr in layer.field_names:
                # Discard payload
                # if str(attr) == "payload":
                #     continue
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
    print(keys)
    for frame_num, frame_info in trace_file_info.items():
        res = ""
        for key in keys:
            if key in frame_info:
                res += frame_info[key] + ","
            else:
                res += "NA,"
        print(res)
    # print(keys)

test_single_trace("/users/KevinHZ/NUS_Capstone/scripts/collect_quic_traces/temp/www_google_com/1_www_google_com/1_www_google_com.pcap")

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
        try:
            for domain_sub_path in os.listdir(domain_path):
                trace_file_path = f"{domain_path}{domain_sub_path}/"
                for trace_file in os.listdir(trace_file_path):
                    if ".pcap" in trace_file:
                        # trace_file = pyshark.FileCapture(trace_file_path + trace_file, display_filter="quic")
                        domain_ips = get_domain_ips(trace_file_path + trace_file)
                        domain_ips.discard(host_ip)
                        # Discard AdGuard DoQ proxy ip
                        domain_ips.discard("94.140.14.14")
                        print(f"-----{trace_file}: {domain_ips} -----")
                        if not domain_ips:
                            continue
                        domain_ips.add("94.140.14.14")
                        frame_nums = get_frame_nums(trace_file_path + trace_file, domain_ips)
                        features = parse_selected_frame(trace_file_path + trace_file, frame_nums)
                        dataset_file_path = cur_domain_dir + trace_file.split("_")[0] + ".csv"
                        print(dataset_file_path)
                        with open(dataset_file_path, mode ='w') as dataset_file:
                            dataset_file.write(",".join(features[0].keys()) + '\n')
                            for feature in features:
                                dataset_file.write(",".join(feature.values()) + '\n')
        except Exception as e:
            print("An error occurred:", e)
            continue
# parse_traces()

# # Store every field for the protocol
# for domain in os.listdir(args.trace_file_dir):
#     # Store host
#     cur_domain = domain.split(".")[0]
#     print(cur_domain)
#     dataset.update({cur_domain : dict()})
#     domain_path = f"{args.trace_file_dir}{cur_domain}/"
#     for trace_file in os.listdir(domain_path):
#         domain_trace_file = pyshark.FileCapture(domain_path + trace_file)
#         try:
#             h3 = False
#             for packet in domain_trace_file:
#                 if "quic" in packet:
#                     print(packet.quic.)
#                     h3 = True
#                     break
#             if h3:
#                 print(trace_file)
#         except:
#             continue
    # try:
    #     for packet in host_trace_file:
    #         if args.protocol in packet:
    #             dataset[cur_host].update({packet.frame_info.number : dict()})
    #             for field_name in getattr(packet, args.protocol).field_names:
    #                 dataset[cur_host][packet.frame_info.number].update({field_name : getattr(getattr(packet, args.protocol), field_name)})
    # except:
    #     continue

# # Get all possible field names
# field_name_set = set()
# for host, frame_info in dataset.items():
#     for frame_num, field_info in frame_info.items():
#         field_name_set.update(field_info.keys())

# # Ensure the same iterating sequence
# field_name_list = sorted(list(field_name_set))

# # Ouput csv file
# with open(args.dataset_trace_file_path, "w") as f:
#     # Write title
#     title = ",".join(field_name_list) + ",label" "\n"
#     f.write(title)
#     for host, frame_info in dataset.items():
#         for field_info in frame_info.values():
#             cur_frame = str()
#             for field_name in field_name_list:
#                 if field_name in field_info:
#                     cur_frame += field_info[field_name] + ","
#                 else:
#                     cur_frame += "NA,"
#             cur_frame += host + "\n"
#             f.write(cur_frame)
