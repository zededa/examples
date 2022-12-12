#!/bin/bash

getIotedgeVersion() {
	version=`iotedge  --version |  cut -d ' ' -f2`
	if [[ $version == "1.0.10"* ]]; then
		runtimeVersion="1.0.10"
	elif [[ $version == "1.0.9"* ]]; then
		runtimeVersion="1.0.19"
	elif [[ $version == "1.0.8"* ]]; then
		runtimeVersion="1.0.8"
	fi
}

editHostName() {
	iotedgeConfig="/etc/iotedge/config.yaml"
	nodeHostName=`hostname`	
	sed -i "s/hostname:\s\"<ADD HOSTNAME HERE>\"/hostname: \"$nodeHostName\"/g" $iotedgeConfig
}
disableManualProvisioning() {
	linenum=`sed -n '/\s*source\:\s*\"manual/{=;p}' /etc/iotedge/config.yaml | head -1`
	firstline=`expr $linenum - 1`
	if [[ $1 == "1.0.10" ]]; then
		lastline=`expr $linenum + 2`
	else
		lastline=`expr $linenum + 1`
	fi
	sed -i "${firstline},${lastline} s/^/#/" $iotedgeConfig
}

enableManualProvisioning() {
	iotedgeConfig="/etc/iotedge/config.yaml"
	sed -i "s|<ADD DEVICE CONNECTION STRING HERE>|$connection_string|" $iotedgeConfig	
}

enableSymmetricKeyProvisioning() {
	iotedgeConfig="/etc/iotedge/config.yaml"
	linenum=`sed -n '/\s*method\:\s*\"symmetric_key/{=;p}' /etc/iotedge/config.yaml | sed '{N;s/\n.*/ /}'`
	firstline=`expr $linenum - 5`
	if [[ $1 == "1.0.10" ]]; then
		lastline=`expr $linenum + 3`
	else
		lastline=`expr $linenum + 2`
	fi
	sed -i "${firstline},${lastline} s/^##* //" $iotedgeConfig
	scope_id_line=`expr $linenum - 2`
	registration_id_line=`expr $linenum + 1`
	sed -i "${scope_id_line}s/{scope_id}/<SCOPE_ID>/" $iotedgeConfig
	sed -i "${registration_id_line}s/{registration_id}/<REGISTRATION_ID>/"  $iotedgeConfig
	sed -i 's/{symmetric_key}/<SYMMETRIC_KEY>/' $iotedgeConfig
	group_key_bytes=$(echo "$dps_shared_key" | base64 --decode | xxd -p -u -c 1000)
	sas_key=$(echo -n "$node_name" | openssl sha256 -mac HMAC -macopt hexkey:"$group_key_bytes" -binary | base64)
	sed -i "${scope_id_line}s|<SCOPE_ID>|$dps_scopeid|" $iotedgeConfig
	sed -i "${registration_id_line}s|<REGISTRATION_ID>|$symmetric_registration_id|" $iotedgeConfig
	sed -i "s|<SYMMETRIC_KEY>|$sas_key|" $iotedgeConfig
}

enableTPMProvisioning() {
	iotedgeConfig="/etc/iotedge/config.yaml"
	linenum=`sed -n '/\s*method\:\s*\"tpm/{=;p}' /etc/iotedge/config.yaml | sed '{N;s/\n.*/ /}'`
	firstline=`expr $linenum - 5`
	if [[ $1 == "1.0.10" ]]; then
		lastline=`expr $linenum + 2`
	else
		lastline=`expr $linenum + 1`
	fi
	sed -i "${firstline},${lastline} s/^##* //" $iotedgeConfig
	scope_id_line=`expr $linenum - 2`
	registration_id_line=`expr $linenum + 1`
	sed -i "${scope_id_line}s/{scope_id}/<SCOPE_ID>/" $iotedgeConfig
	sed -i "${registration_id_line}s/{registration_id}/<REGISTRATION_ID>/" $iotedgeConfig
	sed -i "${scope_id_line}s|<SCOPE_ID>|$tpm_scopeid|" $iotedgeConfig
	sed -i "${registration_id_line}s|<REGISTRATION_ID>|$tpm_registration_id|" $iotedgeConfig
}

enableDeviceCerts(){
	iotedgeConfig="/etc/iotedge/config.yaml"
	sed -i 's/#\s*certificates:/certificates:/' $iotedgeConfig
	sed -i 's/#\s*certificates:/certificates:/' $iotedgeConfig
	sed -i 's/#\s*device_ca_cert:/  device_ca_cert:/' $iotedgeConfig
	sed -i 's/<ADD PATH TO DEVICE CA CERTIFICATE HERE>/<ADD URI TO DEVICE CA CERTIFICATE HERE>/' $iotedgeConfig
	sed -i 's/#\s*device_ca_pk:/  device_ca_pk:/' $iotedgeConfig
	sed -i 's/<ADD PATH TO DEVICE CA PRIVATE KEY HERE>/<ADD URI TO DEVICE CA PRIVATE KEY HERE>/' $iotedgeConfig
	sed -i 's|#   trusted_ca_certs:|  trusted_ca_certs:|' $iotedgeConfig
	sed -i 's/<ADD PATH TO TRUSTED CA CERTIFICATES HERE>/<ADD URI TO TRUSTED CA CERTIFICATES HERE>/' $iotedgeConfig

	sed -i "s|<ADD URI TO DEVICE CA CERTIFICATE HERE>|/etc/iotedge/${node_name}.name_ca-full-chain.cert.pem|" $iotedgeConfig
	sed -i "s|<ADD URI TO DEVICE CA PRIVATE KEY HERE>|/etc/iotedge/${node_name}.name_ca.key.pem|" $iotedgeConfig
	sed -i 's|"<ADD URI TO TRUSTED CA CERTIFICATES HERE>"|"/etc/iotedge/trusted_ca.cert.pem"|' $iotedgeConfig	
}

createDeviceCerts(){
	echo -e "Create and INstalling IoT edge device certificates"
	echo -e "$node_name"
	echo -e "$ca_password"
	iotedgeConfig="/etc/iotedge/config.yaml"
	certPath="/etc/iotedge"
	cd $certPath
	mkdir -p newcerts
	rm index.txt
	touch index.txt
	rm serial
	bash -c 'echo 1000 > serial'
	openssl genrsa -out ${node_name}.name_ca.key.pem 4096
	chmod 444 ${node_name}.name_ca.key.pem
	openssl req -new -sha256 -key ${node_name}.name_ca.key.pem -subj "/CN=$node_name" -out ${node_name}.name_ca.csr
	openssl ca -batch -config openssl_ca.cnf -extensions "v3_intermediate_ca" -days 365 -notext -md sha256 -in ${node_name}.name_ca.csr -cert trusted_ca.cert.pem -keyfile trusted_ca.key.pem -keyform PEM -passin pass:$ca_password -out ${node_name}.name_ca.cert.pem -outdir newcerts

	chmod 444 ${node_name}.name_ca.cert.pem
	cat ${node_name}.name_ca.cert.pem trusted_ca.cert.pem  > ${node_name}.name_ca-full-chain.cert.pem
	chmod 444 ${node_name}.name_ca-full-chain.cert.pem
	cd ~
}

getIotedgeVersion
editHostName
if [ ${provisioning_configuration} ]; then
	if [ ${provisioning_configuration} == 'manual' ]; then
		enableManualProvisioning $runtimeVersion
	elif [ ${provisioning_configuration} == 'tpm' ]; then
		disableManualProvisioning $runtimeVersion
		enableTPMProvisioning $runtimeVersion
	elif [ ${provisioning_configuration} == 'symmetric_key' ]; then
		disableManualProvisioning $runtimeVersion
		enableSymmetricKeyProvisioning $runtimeVersion
	fi
elif [ ${dps_shared_key} == '$webhook.azure.dps.shared_key' ]; then
	disableManualProvisioning $runtimeVersion
	enableTPMProvisioning $runtimeVersion
elif [ ${dps_shared_key} != '$webhook.azure.dps.shared_key' ]; then
	disableManualProvisioning $runtimeVersion
	enableSymmetricKeyProvisioning $runtimeVersion
fi
	

#Installing iotedge gateway certificates
if  [ ${device_certs_required} ]; then 
	if [ ${device_certs_required} == true ]; then
		createDeviceCerts
		enableDeviceCerts
	fi
elif [ -f "/etc/iotedge/trusted_ca.cert.pem" ]; then
	createDeviceCerts
	enableDeviceCerts
fi
