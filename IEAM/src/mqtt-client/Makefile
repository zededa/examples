# Check all necessary environment variables
-include ../../env.check.mk
all: build publish-service

build:
	chmod +x get_broker_ip.sh
	$(shell ./get_broker_ip.sh)
publish-service:
	. ../../env.ibm
	hzn exchange service publish -O -f horizon/mqtt-client-definition.json --pull-image
