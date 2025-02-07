#!/usr/bin/bash
apt-get update
apt-get install -y software-properties-common python3-pip zip wget tcpdump
# Install tshark
add-apt-repository -y ppa:wireshark-dev/stable
DEBIAN_FRONTEND=noninteractive apt-get install -y tshark
usermod -a -G wireshark $USER
# Install chrome
mkdir temp
cd temp
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
DEBIAN_FRONTEND=noninteractive apt-get install -y ./google-chrome-stable_current_amd64.deb
# Install go
wget https://dl.google.com/go/go1.20.linux-amd64.tar.gz
tar -xvf go1.20.linux-amd64.tar.gz
mv go /usr/local
export GOROOT=/usr/local/go
export GOPATH=$HOME/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
echo "export PATH=$GOPATH/bin:$GOROOT/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
cd ..
rm -rf ./temp
# Install dns proxy
go install github.com/AdguardTeam/dnsproxy@latest
# Install python modules
pip3 install argparse pyshark selenium webdriver_manager nest_asyncio pandas numpy scikit-learn
