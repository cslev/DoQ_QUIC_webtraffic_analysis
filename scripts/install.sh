#!/usr/bin/bash
sudo apt-get update
sudo apt-get install -y software-properties-common python3-pip zip
# Install tshark
sudo add-apt-repository -y ppa:wireshark-dev/stable
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tshark
sudo usermod -a -G wireshark $USER
# Install chrome
mkdir temp
cd temp
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ./google-chrome-stable_current_amd64.deb
# Install go
wget https://dl.google.com/go/go1.20.linux-amd64.tar.gz
sudo tar -xvf go1.20.linux-amd64.tar.gz
sudo mv go /usr/local
export GOROOT=/usr/local/go
export GOPATH=$HOME/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
echo "export PATH=$GOPATH/bin:$GOROOT/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
cd ..
sudo rm -rf ./temp
# Install dns proxy
go install github.com/AdguardTeam/dnsproxy@latest
# Install python modules
pip3 install argparse pyshark selenium webdriver_manager nest_asyncio pandas numpy scikit-learn
# Install docker
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
# Config git
git config --global user.name KevinHaoranZhang
git config --global user.email zhr9118@gmail.com
git config --global core.editor vim
