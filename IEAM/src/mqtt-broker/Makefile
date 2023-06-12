# Check all necessary environment variables
-include ../../env.check.mk

export ARCH ?= $(shell hzn architecture)

all: build publish-service

build:
	chmod +x pre-config.sh
	$(shell ./pre-config.sh)
publish-service:
	hzn exchange service publish -O -f horizon/mqtt-definition.json --pull-image



