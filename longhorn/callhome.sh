#!/bin/bash
URL=$1
[ -z $URL ] && echo "failing" && exit 0
while true
do
    curl --insecure -sfL $URL && break
    sleep 5
done
while true
do
    if kubectl get nodes; then
        break
    fi
    sleep 10
done
curl --insecure -sfL $URL | kubectl apply -f -
