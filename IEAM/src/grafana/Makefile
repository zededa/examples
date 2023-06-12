# Check all necessary environment variables
-include ../../env.check.mk
all: publish-service

publish-service:
	hzn exchange service publish -O -f horizon/grafana-definition.json --pull-image

deploy-policy:
	hzn exchange deployment addpolicy -f horizon/grafana-deployment.json mqtt.deployment
