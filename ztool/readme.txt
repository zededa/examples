Ztool is a tool to convert docker-compose files to zededa supported container deployment manifest.
Ztool is a helper tool to create edge application manifest in supported format by zedcontrol. The manifest can be used
in zcli/UI to create an application bundle (edge-app).

Requirements:

1. kompose
2. python3

Ztool leverages kompose to convert docker-compose files to multiple zedControl edge-app deployment formats. Below are
the steps to install kompose on Linux and Windows OS.

Ubuntu:
 wget https://github.com/kubernetes/kompose/releases/download/v1.26.1/kompose_1.26.1_amd64.deb # Replace 1.26.1 with latest tag
 sudo apt install ./kompose_1.26.1_amd64.deb

Mac:
 brew install kompose

ztool supports conversion of V1, V2, and V3 Docker Compose files into zededa edge-app manifest.

Let's take an example of below docker-compose file

file name: docker-compose-multi.yaml
version: '3'

services:
  database:
    image: postgres
    volumes:
      - volume:/var/lib/postgresql
    environment:
      - POSTGRES_DB=beersnobdb, beersnobdb_dev
      - POSTGRES_USER=mhuls
      - POSTGRES_PASSWORD=aStrongPassword
    ports:
      - 54321:5432
  frontend:
    image: nginx
    ports:
      - 30002:80
    environment:
      - NGINX_HOST=foobar.com
      - NGINX_PORT=80

➜  $ python3 ztool.py convert docker-compose-multi.yaml  #command to convert docker compose to kubernetes deployment object
INFO Kubernetes file "database-service.yaml" created
INFO Kubernetes file "frontend-service.yaml" created
INFO Kubernetes file "database-deployment.yaml" created
INFO Kubernetes file "volume-persistentvolumeclaim.yaml" created
INFO Kubernetes file "frontend-deployment.yaml" created

➜  $ python3 ztool.py convertToApp --pod-definition=frontend-deployment.yaml --service-definition=frontend-service.yaml > frontend-manifest.json //to convert frontend application manifest

➜  $ python3 ztool.py convertToApp --pod-definition=database-deployment.yaml --service-definition=database-service.yaml > database-manifest.json. //to convert database application manifest

Execute from docker container:
# Build container
  docker build -t ztool .
# Run container. mount current directory to container, assuming docker compose is present inside PWD
  docker run -it -v $PWD:/tmp ztool
# Execute command to convert docker compose to  POD deployment
  docker exec  -it f95be5681aab  sh -c  "cd /tmp && python /ztool.py convert docker-compose.yaml"

  INFO Kubernetes file "web-service.yaml" created
  INFO Kubernetes file "web-deployment.yaml" created
# Execute command to convert POD deployment to  POD deployment
  docker exec  -it f95be5681aab  sh -c  "cd /tmp && python /ztool.py convertToApp --pod-definition=web-deployment.yaml --service-definition=web-service.yaml" > web-deployment.json

Create edge-app using zcli:

# pull latest zcli container
  docker pull zededa/zcli:latest
# run zcli container
  docker run -it -v $(PWD):/tmp zededa/zcli:latest
# zcli configure to enter enterprise access credentials or token
  docker configure
# login
  zcli login
# create container edge-app using zcli edge-app create - use json file generated from convertToApp
  zcli edge-app create web --manifest=/tmp/web-deployment.json --title=web

Limitation of the tool:

 -> This tool will only create the container deployment manifest.
        -> this will not create any object inside zedcontrol (for example: datastore, image, edge-app).
        to create datastore, image, volume and edge-app inside zedcontrol user need to invoke corresponding zcli command
        to create all these objects.
        -> Make sure to create all above object in same as mentioned in output json.
        -> app-secret is not support. compose with app-secret as env is not supported.










