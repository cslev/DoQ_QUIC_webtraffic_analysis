# tshark commands

# Capture network traffic
sudo tshark -D
touch sample.pcap
sudo chmod o=rw sample.pcap
sudo tshark -i eno33np0 -f "port 80 or port 443 or port 53" -w /users/KevinHZ/NUS_Capstone/scripts/collect_quic_traces/test/google.pcap

# Read captured files
tshark -r /users/KevinHZ/NUS_Capstone/traces/facebook.pcap -Y "(http2)||(dns and tls)||(tls)||(quic)||(http3)||(dns)||(udp)" -T fields -e frame.number -e _ws.col.Time -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e _ws.col.Protocol -e frame.len -e _ws.col.Info -E header=y -E separator="," -E quote=d -E occurrence=f
tshark -r -Y "(http2)||(dns and tls)||(tls)||(quic)||(http3)||(dns)||(udp)" -T fields -e frame.number -e _ws.col.Time -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e _ws.col.Protocol -e frame.len -e _ws.col.Info -E header=y -E separator="," -E quote=d -E occurrence=f
tshark -r /users/KevinHZ/NUS_Capstone/traces/facebook.pcap -Y "(http2)||(dns and tls)||(tls)||(quic)||(http3)||(dns)||(udp)" -V &> NUS_Capstone/facebook.txt

# http3 commands
sudo docker run -it --rm ymuski/curl-http3 curl https://www.youtube.com/ -IL --http3 --resolve youtube.com:443:8.8.8.8

# Run scripts
sudo python3 NUS_Capstone/scripts/packetCapture.py

# ssl key log file
export SSLKEYLOGFILE=/users/KevinHZ/traces/google/ssl-key.log

# Start AdGuard quic dns resolver
sudo /users/KevinHZ/go/bin/dnsproxy -p 53 -u quic://94.140.14.14/ -l 127.0.0.1

# Start packet capture
nohup python3 packetCapture.py --websites_count 1000 --access_count 25 &> output.txt &
nohup python3 packetCapture.py --filtered_target_websites ./open_world_websites.txt --websites_count 10000 --access_count 1 &> output.txt &

# Merge dataset
nohup sudo python3 merge_zip.py --dataset_zip_dir ./dataset_zip/ --output_dataset_path /mydata/dataset/ &> ./output/output_dataset.txt &
nohup sudo python3 merge_zip.py --dataset_zip_dir ./dataset_open_zip/ --output_dataset_path /mydata/dataset_open/ &> ./output/output_dataset_open.txt &
nohup sudo python3 merge_zip.py --dataset_zip_dir ./dataset_utah_zip/ --output_dataset_path /mydata/dataset_utah/ &> output_utah.txt &
nohup sudo python3 merge_zip.py --dataset_zip_dir ./dataset_wisc_zip/ --output_dataset_path /mydata/dataset_wisc/ &> output_wisc.txt &
nohup sudo python3 merge_zip.py --dataset_zip_dir ./dataset_mass_zip/ --output_dataset_path /mydata/dataset_mass/ &> output_mass.txt &
nohup sudo python3 merge_zip.py --dataset_zip_dir ./dataset_clem_zip/ --output_dataset_path /mydata/dataset_clem/ &> output_clem.txt &

# Start npz file parsing
# Close-world npz
nohup python3 csv_to_npz.py --num_classes 100 --sequence_len 200 --output_file_path ./dataset_100_200.npz &> output100.txt &
nohup python3 csv_to_npz.py --num_classes 200 --sequence_len 200 --output_file_path ./dataset_200_200.npz &> output200.txt &
nohup python3 csv_to_npz.py --num_classes 300 --sequence_len 200 --output_file_path ./dataset_300_200.npz &> output300.txt &
nohup python3 csv_to_npz.py --num_classes 400 --sequence_len 200 --output_file_path ./dataset_400_200.npz &> output400.txt &
nohup python3 csv_to_npz.py --num_classes 500 --sequence_len 200 --output_file_path ./dataset_500_200.npz &> output500.txt &

nohup python3 csv_to_npz.py --num_classes 200 --sequence_len 200 --output_file_path ./dataset_200_utah.npz --dataset_dir /mydata/dataset_utah/ &> output200_utah.txt &
nohup python3 csv_to_npz.py --num_classes 200 --sequence_len 200 --output_file_path ./dataset_200_wisc.npz --dataset_dir /mydata/dataset_wisc/ &> output200_wisc.txt &
nohup python3 csv_to_npz.py --num_classes 200 --sequence_len 200 --output_file_path ./dataset_200_mass.npz --dataset_dir /mydata/dataset_mass/ &> output200_mass.txt &
nohup python3 csv_to_npz.py --num_classes 200 --sequence_len 200 --output_file_path ./dataset_200_clem.npz --dataset_dir /mydata/dataset_clem/ &> output200_clem.txt &

nohup python3 csv_to_npz.py --num_classes 500 --sequence_len 200 --output_file_path ./dataset_500_utah.npz --dataset_dir /mydata/dataset_utah/ &> output500_utah.txt &
nohup python3 csv_to_npz.py --num_classes 500 --sequence_len 200 --output_file_path ./dataset_500_wisc.npz --dataset_dir /mydata/dataset_wisc/ &> output500_wisc.txt &
nohup python3 csv_to_npz.py --num_classes 500 --sequence_len 200 --output_file_path ./dataset_500_mass.npz --dataset_dir /mydata/dataset_mass/ &> output500_mass.txt &
nohup python3 csv_to_npz.py --num_classes 500 --sequence_len 200 --output_file_path ./dataset_500_clem.npz --dataset_dir /mydata/dataset_clem/ &> output500_clem.txt &

nohup python3 csv_to_npz.py --num_classes 100 --sequence_len 400 --output_file_path ./dataset_npz/dataset_100_400.npz &> ./output/output100_400.txt &
nohup python3 csv_to_npz.py --num_classes 100 --sequence_len 600 --output_file_path ./dataset_npz/dataset_100_600.npz &> ./output/output100_600.txt &
nohup python3 csv_to_npz.py --num_classes 100 --sequence_len 800 --output_file_path ./dataset_npz/dataset_100_800.npz &> ./output/output100_800.txt &
nohup python3 csv_to_npz.py --num_classes 100 --sequence_len 1000 --output_file_path ./dataset_npz/dataset_100_1000.npz &> ./output/output100_1000.txt &

# Open-world npz
nohup python3 csv_to_npz.py --sequence_len 200 --ow --num_classes 100 --num_datapoints 200  --num_classes_ow 25000 --num_datapoints_ow 4 --output_file_path ./dataset_npz/dataset_100_200_25k_4_ow.npz &> ./output/output_100_200_25k_4_ow.txt &
nohup python3 csv_to_npz.py --sequence_len 200 --ow --num_classes 100 --num_datapoints 450  --num_classes_ow 47500 --num_datapoints_ow 4 --split_ratio 0.8 --split_ratio_ow 0.5 --output_file_path ./dataset_npz/dataset_100_450_47500_4_ow.npz &> ./output/output_100_450_47500_4_ow.txt &

# Get dataset statistics
nohup python3 get_dataset_stats.py --num_classes 100 --sequence_len 200 --output_file_path ./dataset_stats/dataset_stats_100_200.csv &> ./output/dataset_stats_100_200.txt &
nohup python3 get_dataset_stats.py --num_classes 100 --sequence_len 400 --output_file_path ./dataset_stats/dataset_stats_100_400.csv &> ./output/dataset_stats_100_400.txt &
nohup python3 get_dataset_stats.py --num_classes 100 --sequence_len 600 --output_file_path ./dataset_stats/dataset_stats_100_600.csv &> ./output/dataset_stats_100_600.txt &
nohup python3 get_dataset_stats.py --num_classes 100 --sequence_len 800 --output_file_path ./dataset_stats/dataset_stats_100_800.csv &> ./output/dataset_stats_100_800.txt &
nohup python3 get_dataset_stats.py --num_classes 100 --sequence_len 1000 --output_file_path ./dataset_stats/dataset_stats_100_1000.csv &> ./output/dataset_stats_100_1000.txt &
nohup python3 get_dataset_stats.py --num_classes 100 --parse_all --output_file_path ./dataset_stats/dataset_100_all.csv &> ./output/dataset_stats_100_all.txt &

# Get canadidate open world websites
python3 filter_open_world.py --num_classes 10000 --num_datapoints 4

# SoC GPU
sbatch config_soc.sh
