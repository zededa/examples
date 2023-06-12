#!/bin/bash

mkdir -p /home/pocuser
cat >/home/pocuser/mosquitto.conf <<EOL
persistence true
persistence_location /mosquitto/data/
log_dest file /mosquitto/log/mosquitto.log
EOL
chmod 777 /home/pocuser/mosquitto.conf
