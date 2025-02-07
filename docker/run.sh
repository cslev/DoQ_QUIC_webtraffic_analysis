cd docker
chmod +x setup_docker_chrome.sh
./setup_docker_chrome.sh
sudo docker build -t myubuntu:latest .
sudo docker run -d --shm-size=4gb --name instance1 myubuntu:latest
sudo docker run -d --shm-size=4gb --name instance2 myubuntu:latest
