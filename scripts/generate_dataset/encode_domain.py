#!/usr/bin/python3
import pandas as pd
import os

def encode_domain_int(filtered_target_websites, dataset_path, output_path):
    domain_to_index = dict()
    index = 1
    min_data, max_data = float("inf"), float("-inf")
    domain_count = len([f for f in os.listdir(dataset_path)])
    print(domain_count)
    with open(filtered_target_websites) as target_websites:
        for info in target_websites:
            url = info.split(";")[0]
            domain = url.split("//")[1]
            dir_name = "_".join(domain.split("."))
            dir_name = "_".join(dir_name.split("/"))[:-1]
            # print(domain)
            domain_path = f"{dataset_path}{dir_name}"
            data_point = len([f for f in os.listdir(domain_path)])
            min_data, max_data = min(min_data, data_point), max(max_data, data_point)
            if os.path.exists(domain_path):
                if dir_name not in domain_to_index:
                    domain_to_index[dir_name] = index
                    index += 1
            if index > domain_count:
                break
    print(min_data, max_data)
    with open(output_path, mode ='w') as output_file:
        output_file.write("url;index\n")
        for k, v in domain_to_index.items():
            print(k, v)
            output_file.write(k + ";" + str(v) + '\n')

def check_domain_match(dataset_path, domain_index_mapping_path):
    df = pd.read_csv(domain_index_mapping_path, sep=';', header=0)
    domain_to_index = df.set_index("url")["index"].to_dict()
    print(domain_to_index)
    for domain in os.listdir(dataset_path):
        if domain not in domain_to_index:
            print(domain)

if __name__ == '__main__':
    encode_domain_int("../get_target_websites/filtered_target_websites.txt", "/mydata/dataset/", "./domain_index_mapping.csv")
    # check_domain_match("/mydata/dataset_10_w2/", "./domain_index_mapping.csv")
