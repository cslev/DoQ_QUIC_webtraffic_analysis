#!/usr/bin/python3
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser(description="Produce dataset in numpy array format.")
parser.add_argument("--num_classes", dest="num_classes", default=986, type=int,
                    help="number of monitored websites to be parsed")
parser.add_argument("--num_classes_ow", dest="num_classes_ow", default=10000, type=int,
                    help="number of unmonitored websites to be parsed")
parser.add_argument("--num_datapoints", dest="num_datapoints", default=2000, type=int,
                    help="number of datapoins to keep per class in monitored set")
parser.add_argument("--num_datapoints_ow", dest="num_datapoints_ow", default=4, type=int,
                    help="number of datapoins to keep per class in unmonitored set")
parser.add_argument("--split_ratio", dest="split_ratio", default=0.8, type=float,
                    help="train and test split ratio for monitored set")
parser.add_argument("--split_ratio_ow", dest="split_ratio_ow", default=0.5, type=float,
                    help="train and test split ratio for unmonitored set")
parser.add_argument("--sequence_len", dest="sequence_len", default=200, type=int,
                    help="sequence length for each website trace")
parser.add_argument("--dataset_dir", dest="dataset_dir", default="/mydata/dataset/",
                    help="dataset csv files directory")
parser.add_argument("--dataset_ow_dir", dest="dataset_ow_dir", default="/mydata/dataset_open/",
                    help="open world dataset csv files directory")
parser.add_argument('--ow', dest="ow", action='store_true', default=False,
                    help='enable open world dataset parsing')
parser.add_argument("--output_file_path", dest="output_file_path", default="./dataset.npz",
                    help="output .npz file directory")
args = parser.parse_args()

def parse_single_file(input_file_path, index_range: int = 200):
    df = pd.read_csv(input_file_path, sep=';', header=0)
    # Get host ip
    host_ip = str(df.loc[df['dst_ip'] == "94.140.14.14"].iloc[0]["src_ip"])
    host_ip
    # Set Protocol: DoQ, QUIC, TCP with one hot encoding
    def check_protocol(df):
        if df['dst_ip'] == "94.140.14.14" or df['src_ip'] == "94.140.14.14":
            return "DoQ"
        if df['protocol'] == 1:
            return "QUIC"
        return "TCP"
    protocols = df.apply(check_protocol, axis=1)
    protocols_onehot = pd.get_dummies(protocols)
    if "DoQ" not in protocols_onehot:
        protocols_onehot["DoQ"] = 0
    if "QUIC" not in protocols_onehot:
        protocols_onehot["QUIC"] = 0
    if "TCP" not in protocols_onehot:
        protocols_onehot["TCP"] = 0
    protocols_onehot = protocols_onehot[["DoQ", "QUIC", "TCP"]]
    scaler = StandardScaler()
    # Standarized inter arrival time and packet size
    df[['iat_std', 'len_std']] = scaler.fit_transform(df[['relative_time', 'length']])
    # Set directions: in(-1), out(1)
    def check_direction(df):
        if df['src_ip'] == host_ip:
            return 1
        return -1
    directions = df.apply(check_direction, axis=1)
    index = 0
    sequence = []
    while index < min(index_range, len(df) - 1):
        sequence.append([int(protocols_onehot.iloc[index].values[0]), 1 if int(protocols_onehot.iloc[index].values[1]) else 0, df['length'][index], df['relative_time'][index], directions[index]])
        index += 1
    while index < index_range:
        sequence.append([0, 0, 0, 0, 0])
        index += 1
    # Sequnence: DoQ (1 if DoQ), TCP/QUIC(1 if QUIC), Inter-Arrival Time (Standarized), Packet Length (Standarized), Direction (-1 if IN, 1 if OUT)
    # return sequence
    return np.array(sequence)

def parse_all_files(dataset_path, domain_index_mapping_path, num_classes: int = 1000, num_datapoints: int = 2000):
    x, y = [], []
    domain_count = 0
    # Get domain to index mapping
    df = pd.read_csv(domain_index_mapping_path, sep=';', header=0)
    domain_to_index = df.set_index("url")["index"].to_dict()
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
                x.append(sequence)
                y.append(index)
                datapoints_count += 1
            except:
                print(f"Error parsing file: {input_file_path}")
                continue
    return np.array(x), np.array(y)

def parse_all_files_ow(dataset_path,domain_index_mapping_path, class_index, num_classes: int = 10000, num_datapoints: int = 4):
    x, y = [], []
    domain_count = 0
    # Get domain to index mapping
    df = pd.read_csv(domain_index_mapping_path, sep=';', header=0)
    domain_to_index = df.set_index("url")["index"].to_dict()
    for domain in os.listdir(dataset_path):
        if domain_count == num_classes:
            break
        if domain in domain_to_index:
            continue
        input_domain_path = f"{dataset_path}{domain}/"
        file_count = len([f for f in os.listdir(input_domain_path)])
        if file_count < num_datapoints:
            continue
        domain_count += 1
        print(f"{domain}: {file_count} {class_index}")
        for i in range(1, num_datapoints + 1):
            file_dir = f"{i}_{domain}"
            input_file_path = f"{input_domain_path}{file_dir}.csv"
            try:
                sequence = parse_single_file(input_file_path, args.sequence_len)
                x.append(sequence)
                y.append(class_index)
                print(f"parsed file {input_file_path}")
            except:
                print(f"Error parsing file: {input_file_path}")
                continue
    return np.array(x), np.array(y)

if __name__ == '__main__':
    domain_index_mapping_path = "./domain_index_mapping.csv"
    x, y = parse_all_files(args.dataset_dir, domain_index_mapping_path, args.num_classes, args.num_datapoints)
    print(x.shape)
    print(y.shape)
    # Perform uniform train and test splits
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Perform per class train and test splits
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - args.split_ratio), random_state=42)
    x_train, x_test, y_train, y_test = [], [], [], []
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    # Add open world data
    if args.ow:
        x_ow, y_ow = parse_all_files_ow(args.dataset_ow_dir, domain_index_mapping_path, args.num_classes + 1, args.num_classes_ow, args.num_datapoints_ow)
        print(x_ow.shape)
        print(y_ow.shape)
        dataset_ow_len = x_ow.shape[0]
        ow_split = int(dataset_ow_len * args.split_ratio_ow)
        x_ow_train, x_ow_test = x_ow[:ow_split], x_ow[ow_split:]
        y_ow_train, y_ow_test = y_ow[:ow_split], y_ow[ow_split:]
        x_train = np.concatenate((x_train, x_ow_train), axis=0)
        y_train = np.concatenate((y_train, y_ow_train), axis=0)
        x_test = np.concatenate((x_test, x_ow_test), axis=0)
        y_test = np.concatenate((y_test, y_ow_test), axis=0)
        print("ow dataste info")
        print(x_ow_train.shape)
        print(y_ow_train.shape)
        print(x_ow_test.shape)
        print(y_ow_test.shape)
        print("after merge")
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
    np.savez_compressed(args.output_file_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    # Check saved npz file
    # dict_data = np.load(args.output_file_path)
    # x_train, x_test, y_train, y_test = dict_data['x_train'], dict_data['x_test'], dict_data['y_train'], dict_data['y_test']
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # print(np.unique(y_test))

    # Test single file parsing
    # sequence = parse_single_file("./dataset/www_flightradar24_com/446_www_flightradar24_com.csv")
    # for line in sequence:
    #     print(line)
