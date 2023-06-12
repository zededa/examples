-include env.check.mk
export ARCH ?= $(shell hzn architecture)
ifeq ($(ARCH),arm64)
	$(error container image in this deployment is only supported on X86_64)
endif

all: publish-all

publish-all: publish-mqttbroker publish-mqttclient publish-mqttclient publish-grafana

publish-mqttbroker:
	make -C src/mqtt-broker build
	make -C src/mqtt-broker publish-service

publish-mqttsimulator:
	make -C src/mqtt-simulator build
	make -C src/mqtt-simulator push
	make -C src/mqtt-simulator publish-service

publish-mqttclient:
	make -C src/mqtt-client build
	make -C src/mqtt-client publish-service

publish-grafana:
	make -C src/grafana publish-service

deploy-policy-all:
	make -C src/grafana config-deploy-policy
