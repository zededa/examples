# Check all necessary environment variables
-include ../../env.check.mk

all: publish-service

publish-service:
	hzn exchange service publish -O -f horizon/mqtt-simulator-definition.json --pull-image

