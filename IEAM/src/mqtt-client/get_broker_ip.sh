#!/bin/bash
ip_addr=`ip addr show $(ip route | awk '/default/ { print $5 }') | grep "inet" | head -n 1 | awk '/inet/ {print $2}' | cut -d'/' -f1`
sed -i -e "s/localhost/${ip_addr}/g" ../../env.ibm
source ../../env.ibm

