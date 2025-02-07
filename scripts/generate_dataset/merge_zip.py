#!/usr/bin/python3
import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description="Merge zipped dataset into one.")
parser.add_argument("--dataset_zip_dir", dest="dataset_zip_dir", default="./dataset_zip/",
                    help="zipped dataset csv files directory")
parser.add_argument("--output_dataset_path", dest="output_dataset_path", default="/mydata/dataset/",
                    help="output summary dataset file directory")
args = parser.parse_args()

def merge(input_path, output_path):
    for domain in os.listdir(input_path):
        domain_index = 1
        output_domain_path = f"{output_path}{domain}/"
        if os.path.exists(output_domain_path):
            # Update file index
            domain_index = len([f for f in os.listdir(output_domain_path)]) + 1
        else:
            os.mkdir(output_domain_path)
        input_domain_path = f"{input_path}{domain}/"
        print(f"--- moving {input_domain_path}")
        file_count = len([f for f in os.listdir(input_domain_path)])
        for i in range(1, file_count + 1):
            file_dir = f"{i}_{domain}"
            input_file_path = f"{input_domain_path}{file_dir}.csv"
            output_file_path = f"{output_domain_path}{domain_index}_{domain}.csv"
            domain_index += 1
            print(input_file_path)
            print(output_file_path)
            if os.path.exists(input_file_path):
                print(f"!!! mv {input_file_path} {output_file_path}")
                subprocess.run(f"mv {input_file_path} {output_file_path}", shell=True)

def merge_all(dataset_zip_dir, dataset_path):
    for zip_path in os.listdir(dataset_zip_dir):
        file_name = zip_path.split(".")[0]
        zip_file_path = f"{dataset_zip_dir}{zip_path}"
        unzip_file_path = f"./{file_name}/"
        print(f"!!! unzip {zip_file_path}")
        subprocess.run(f"unzip {zip_file_path}", shell=True)
        merge(unzip_file_path, dataset_path)
        print(f"!!! rm -rf {unzip_file_path}")
        subprocess.run(f"rm -rf {unzip_file_path}", shell=True)

if __name__ == '__main__':
    dataset_path = args.output_dataset_path
    if os.path.exists(dataset_path):
        subprocess.run(f"rm -rf {dataset_path}", shell=True)
    os.mkdir(dataset_path)
    merge_all(args.dataset_zip_dir, dataset_path)
