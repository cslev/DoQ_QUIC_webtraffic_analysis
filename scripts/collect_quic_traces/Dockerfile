FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip software-properties-common wget
# Install tshark
RUN add-apt-repository -y ppa:wireshark-dev/stable
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tshark
# Install chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ./google-chrome-stable_current_amd64.deb
# Install go
RUN wget https://dl.google.com/go/go1.20.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go1.20.linux-amd64.tar.gz
ENV PATH /usr/local/go/bin:$PATH
# Install dns proxy
RUN go install github.com/AdguardTeam/dnsproxy@latest
# Install python modules
RUN pip3 install argparse pyshark selenium webdriver_manager
COPY packetCapture.py ./
COPY target_websites.txt ./

CMD ["python3", "./packetCapture.py"]
