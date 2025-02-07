#!/usr/bin/python3
import argparse
import pandas as pd
import numpy as np
import os
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser(description="Produce dataset in numpy array format.")
parser.add_argument("--num_classes", dest="num_classes", default=1000, type=int,
                    help="number of websites to be parsed")
parser.add_argument("--num_datapoints", dest="num_datapoints", default=2000, type=int,
                    help="number of datapoins to keep per class")
parser.add_argument("--sequence_len", dest="sequence_len", default=200, type=int,
                    help="sequence length for each website trace")
parser.add_argument("--dataset_dir", dest="dataset_dir", default="/mydata/dataset/",
                    help="dataset csv files directory")
parser.add_argument("--output_file_path", dest="output_file_path", default="./dataset_stats.csv",
                    help="output statistics file directory")
parser.add_argument('--parse_all', dest="parse_all", action='store_true', default=False,
                    help='parse all packets regardless of sequence length')
args = parser.parse_args()

def parse_single_file(input_file_path, index_range: int = 200):
    df = pd.read_csv(input_file_path, sep=';', header=0)
    # Cut dataframe to specified sequence length
    if not args.parse_all:
        df = df.head(min(len(df), index_range))
    # Get host ip
    host_ip = str(df.loc[df['dst_ip'] == "94.140.14.14"].iloc[0]["src_ip"])
    # Set Protocol: DoQ, QUIC, TCP with one hot encoding
    def check_protocol(df):
        if int(df['dst_port']) == 853 or int(df['src_port']) == 853:
            return "DoQ"
        if df['protocol'] == 1:
            return "QUIC"
        return "TCP"
    protocols = df.apply(check_protocol, axis=1)
    doq_len = len(protocols[protocols == "DoQ"])
    quic_len = len(protocols[protocols == "QUIC"])
    tcp_len = len(protocols[protocols == "TCP"])
    total_len = len(df)
    first_doq = 0 if len(protocols[protocols == 'DoQ']) == 0 else protocols[protocols == 'DoQ'].idxmax() + 1
    last_doq = 0 if len(protocols[protocols == 'DoQ']) == 0 else protocols[protocols == 'DoQ'].last_valid_index() + 1
    first_quic = 0 if len(protocols[protocols == 'QUIC']) == 0 else protocols[protocols == 'QUIC'].idxmax() + 1
    last_quic = 0 if len(protocols[protocols == 'QUIC']) == 0 else protocols[protocols == 'QUIC'].last_valid_index() + 1
    first_tcp = 0 if len(protocols[protocols == 'TCP']) == 0 else protocols[protocols == 'TCP'].idxmax() + 1
    last_tcp = 0 if len(protocols[protocols == 'TCP']) == 0 else protocols[protocols == 'TCP'].last_valid_index() + 1
    doq_pert = []
    for seq_len in [200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000, 20000]:
        df_new = df.head(min(len(df), seq_len))
        protocols_new = df_new.apply(check_protocol, axis=1)
        doq_pert.append(len(protocols_new[protocols_new == "DoQ"]))
    # DoQ Length, QUIC length, TCP length, total length, first/last DoQ index, first/last QUIC index, fisrt/last TCP index
    return [doq_len, quic_len, tcp_len, total_len, first_doq, last_doq, first_quic, last_quic, first_tcp, last_tcp] + doq_pert

def parse_all_files(dataset_path, domain_index_mapping_path, num_classes: int = 1000, num_datapoints: int = 2000):
    with open(args.output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["DoQ_len", "QUIC_len", "TCP_len", "total_len", "first_DoQ", "last_DoQ", "first_QUIC", "last_QUIC", "first_TCP", "Last_TCP", "DoQ_200", "DoQ_400", "DoQ_600", "DoQ_800", "DoQ_1000", "DoQ_2000", "DoQ_4000", "DoQ_6000", "DoQ_8000", "DoQ_10000", "DoQ_20000"])
        domain_count = 0
        # Get domain to index mapping
        df = pd.read_csv(domain_index_mapping_path, sep=';', header=0)
        domain_to_index = df.set_index("url")["index"].to_dict()
        sequences = []
        for domain, index in domain_to_index.items():
            if domain_count == num_classes:
                break
            domain_count += 1
            input_domain_path = f"{dataset_path}{domain}/"
            file_count = len([f for f in os.listdir(input_domain_path)])
            print(f"{domain}: {file_count} {index}")
            datapoints_count = 0
            for i in range(1, file_count):
                if datapoints_count == num_datapoints:
                    break
                file_dir = f"{i}_{domain}"
                input_file_path = f"{input_domain_path}{file_dir}.csv"
                try:
                    sequence = parse_single_file(input_file_path, args.sequence_len)
                    datapoints_count += 1
                    file.write(";".join(list(map(str, sequence))) + "\n")
                except Exception as error:
                    print(f"Error parsing file: {input_file_path} {error}")
                    continue

if __name__ == '__main__':
    domain_index_mapping_path = "./domain_index_mapping.csv"
    # sequence = parse_single_file("/mydata/dataset/hackerdoslot_com/1273_hackerdoslot_com.csv", args.sequence_len)
    # print(sequence)
    parse_all_files(args.dataset_dir, domain_index_mapping_path, args.num_classes, args.num_datapoints)
    # with open(args.output_file_path, mode='w', newline='') as file:
    #     writer.writerows(sequences)
