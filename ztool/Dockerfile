FROM alpine:3.13
RUN apk add --no-cache curl
COPY ./kompose_version version
RUN version=$(cat version) && curl -L "https://github.com/kubernetes/kompose/releases/download/v${version}/kompose-linux-amd64" -o kompose
RUN cp -rf ./kompose /usr/bin/kompose
RUN chmod +x /usr/bin/kompose
RUN apk add --no-cache python3 py3-pip && ln -sf python3 /usr/bin/python
RUN pip3 install pyyaml
RUN pip3 install docopt
copy ./ztool.py ztool.py
CMD ["/bin/sh"]
