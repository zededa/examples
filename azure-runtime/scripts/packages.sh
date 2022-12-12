DEV_PACKAGES="
build-essential
curl
git
net-tools
vim
cloud-init
libprotobuf-dev
libssl-dev
libcurl4-openssl-dev
uuid-dev
libprotoc-dev
"
sudo apt update
sudo apt-get -y install $DEV_PACKAGES
curl https://packages.microsoft.com/config/ubuntu/18.04/multiarch/prod.list > ./microsoft-prod.list
sudo cp ./microsoft-prod.list /etc/apt/sources.list.d/
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo cp ./microsoft.gpg /etc/apt/trusted.gpg.d/
#sudo wget https://github.com/cshari-zededa/lfedge-eve-tools/releases/download/v2.0.0/cshari-zededa.list
#sudo cp cshari-zededa.list /etc/apt/sources.list.d/
sudo apt-get update
