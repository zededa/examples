#!/bin/bash

createUserGroup() {

	iotedgeDir="/var/lib/aziot/edged"
	keyStore="/var/lib/aziot/keyd"
	tpmd="/var/lib/aziot/tpmd"
	certd="/var/lib/aziot/certd"
	identityd="/var/lib/aziot/identityd"

	#Create group
	groupadd -g 994 iotedge || true
	groupadd -g 995 aziotid || true
	groupadd -g 996 aziotcs || true
	groupadd -g 997 aziottpm || true
	groupadd -g 998 aziotks || true

	#Creatr users
	useradd -g iotedge -u 994 -d $iotedgeDir --shell=/sbin/nologin iotedge || true
	mkdir -p $iotedgeDir
	useradd -g aziotid -u 995 -d $identityd --shell=/sbin/nologin aziotid
	mkdir -p $identityd
	useradd -g aziotcs -u 996 -d $certd --shell=/sbin/nologin aziotcs
	mkdir -p $certd
	useradd -g aziottpm -u 997 -d $tpmd --shell=/sbin/nologin aziottpm
	mkdir -p $tpmd
	useradd -g aziottpm -u 998 -d $keyStore --shell=/sbin/nologin aziotks
	mkdir -p $keyStore

	#Add user to group 
	usermod -a -G docker iotedge || true
	usermod -a -G iotedge pocuser || true
	chown -R iotedge.iotedge $iotedgeDir
	
	usermod -a -G aziotid iotedge || true
	chown -R aziotid.aziotid $identityd

	usermod -a -G aziotcs aziotid || true
	usermod -a -G aziotcs iotedge || true
	chown -R aziotcs.aziotcs $certd

	usermod -a -G aziottpm	aziotid || true
	chown -R aziottpm.aziottpm $tpmd
	
	usermod -a -G aziotks aziotcs || true
	usermod -a -G aziotks aziotid || true
	usermod -a -G aziotks iotedge || true
	chown -R aziotks.aziotks $keyStore
	
}


postEdgeConfig() {

	iotedgeSocketMgmt="/zed/data2/aziot-edge/lib/systemd/system/aziot-edged.mgmt.socket"
	iotedgeService="/zed/data2/aziot-edge/lib/systemd/system/aziot-edged.service"
	iotedgeWorkloadSocket="/zed/data2/aziot-edge/lib/systemd/system/aziot-edged.workload.socket"
	systemDirMgm="/etc/systemd/system/sockets.target.wants/aziot-edged.mgmt.socket"
	systemDMultiUser="/etc/systemd/system/multi-user.target.wants/aziot-edged.service"
	systemDFile="/etc/systemd/system/sockets.target.wants/aziot-edged.workload.socket"
	iotedgeConfig="/zed/data2/aziot-edge/etc/aziot/config.toml"

    if [ -f "/zed/data2/aziot-edge/etc/aziot/config.toml.edge.template" ];
	then
		cp /zed/data2/aziot-edge/etc/aziot/config.toml.edge.template $iotedgeConfig
	fi

	if [ -d "/zed/data2/aziot-edge/lib/systemd/system" ];
	then
		cp -rf $iotedgeConfig /etc/aziot/config.toml
		rm -rf /lib/systemd/system/aziot-edged.mgmt.socket
		cp -rf $iotedgeSocketMgmt /lib/systemd/system/aziot-edged.mgmt.socket
		ln -s /lib/systemd/system/aziot-edged.mgmt.socket $systemDirMgm
		rm -rf /lib/systemd/system/aziot-edged.service
		cp -rf $iotedgeService /lib/systemd/system/aziot-edged.service
		ln -s /lib/systemd/system/aziot-edged.service $systemDMultiUser
		rm -rf /lib/systemd/system/aziot-edged.workload.socket
		cp -rf $iotedgeWorkloadSocket /lib/systemd/system/aziot-edged.workload.socket
		ln -s /lib/systemd/system/aziot-edged.workload.socket $systemDFile
	fi
	
	#systemctl daemon-reload
	systemctl enable aziot-identityd.socket
	systemctl enable aziot-certd.socket
	systemctl enable aziot-keyd.socket
	systemctl enable aziot-tpmd.socket
	systemctl enable aziot-identityd.service
	systemctl enable aziot-edged
}

postIdentityService(){

	certSocket="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-certd.socket"
	identitySocket="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-identityd.socket"
	keydSocket="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-keyd.socket"
	tpmdSocket="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-tpmd.socket"
	sysCertSocket="/etc/systemd/system/sockets.target.wants/aziot-certd.socket"
	sysIdentitySocket="/etc/systemd/system/sockets.target.wants/aziot-identityd.socket"
	sysKeydSocket="/etc/systemd/system/sockets.target.wants/aziot-keyd.socket"
	sysTpmdSocket="/etc/systemd/system/sockets.target.wants/aziot-tpmd.socket"
	certService="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-certd.service"
	identityService="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-identityd.service"
	keydService="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-keyd.service"
	tpmdService="/zed/data2/aziot-identity-service/lib/systemd/system/aziot-tpmd.service"

	if [ -d "/zed/data2/aziot-identity-service/lib/systemd/system" ];
	then
		rm -rf /lib/systemd/system/aziot-certd.socket
		cp -rf $certSocket /lib/systemd/system/aziot-certd.socket
		ln -s /lib/systemd/system/aziot-certd.socket $sysCertSocket
		rm -rf /lib/systemd/system/aziot-identityd.socket
		cp -rf $identitySocket /lib/systemd/system/aziot-identityd.socket
		ln -s /lib/systemd/system/aziot-identityd.socket $sysIdentitySocket
		rm -rf /lib/systemd/system/aziot-keyd.socket
		cp -rf $keydSocket /lib/systemd/system/aziot-keyd.socket
		ln -s /lib/systemd/system/aziot-keyd.socket $sysKeydSocket
		rm -rf /lib/systemd/system/aziot-tpmd.socket
		cp -rf $tpmdSocket /lib/systemd/system/aziot-tpmd.socket
		ln -s /lib/systemd/system/aziot-tpmd.socket $sysTpmdSocket
		rm -rf /lib/systemd/system/aziot-identityd.service
		cp -rf $identityService /lib/systemd/system/aziot-identityd.service
		rm -rf /lib/systemd/system/aziot-keyd.service
		cp -rf $keydService /lib/systemd/system/aziot-keyd.service
		rm -rf /lib/systemd/system/aziot-certd.service
		cp -rf $certService /lib/systemd/system/aziot-certd.service
		rm -rf /lib/systemd/system/aziot-tpmd.service
		cp -rf $tpmdService /lib/systemd/system/aziot-tpmd.service
	fi

}

walkDir() {
    packagePath=$1
    dataPath=$2
    for pathName in "$packagePath"/*;
	do
		if [ -d $pathName ]; then
            folderName=`echo $pathName | sed "s~${dataPath}~~"`
            if [ ! -d $folderName ];
            then
                mkdir -p "$folderName"
            fi
            walkDir $pathName $dataPath
        elif [ -f $pathName ]; then
            fileName=`echo $pathName | sed "s~${dataPath}~~"`
            ln -s $pathName $fileName
        fi
    done
}

linkToRoot() {
	echo "=====================$1============="
	case "$1" in
		aziot-edge)
			iotedgePackage="/zed/data2/aziot-edge"
			walkDir $iotedgePackage $iotedgePackage
			;;
		aziot-identity-service)
			libiothsmStdPackage="/zed/data2/aziot-identity-service"
			walkDir $libiothsmStdPackage $libiothsmStdPackage
			;;
		lfedge-eve-tools)
			lfedgeEveTools="/zed/data2/lfedge-eve-tools"
			walkDir $lfedgeEveTools $lfedgeEveTools
            cp -rf /zed/data2/lfedge-eve-tools/usr/libexec/aziot-identity-service/aziotd /zed/data2/aziot-identity-service/usr/libexec/aziot-identity-service/
			#ln -s /zed/data2/lfedge-eve-tools/usr/lib/libiothsm.so.1.0.8 /usr/lib/libiothsm.so.1
			#ln -s /zed/data2/lfedge-eve-tools/usr/lib/libiothsm.so.1.0.8 /usr/lib/libiothsm.so
			;;
		*)
			echo "not a valid package"
	esac
}

downloadInstall() {
	packageNameVersion=$1
	dstFolder=$2
	packageLocation="/zed/data2/${dstFolder}"
	if [ ! -d $packageLocation ]; then
		mkdir -p $packageLocation
	fi
	cd $packageLocation
	echo "before download"
	if [[ $dstFolder == "lfedge-eve-tools" ]]; then
		#download_url="https://github.com/cshari-zededa/lfedge-eve-tools/releases/download/v3.0.0/lfedge-eve-tools.deb"
        	download_url="http://10.216.2.82/images/lfedge-eve-tools.deb"
		wget $download_url
	else
		apt download $packageNameVersion
		if [ ${?} -ne 0 ];
		then
			echo "Download ${packageNameVersion} failed"
			exit 2
		fi
	fi
	FILES=($(ls -ls))
	for FILE in "${FILES[@]}";
	do
		if [[ $FILE == *.deb ]];
		then
			echo "deb package name : ${FILE}"
			dpkg-deb -x ${FILE} .
			rm -rf $FILE
		fi
	done
}

declare -a packageList
createUserGroup
packageList=("aziot-edge" "aziot-identity-service" "lfedge-eve-tools")
for package in ${packageList[@]};
do 
	echo "$package"
	if  [[ $package == aziot-edge* ]];then
		packageName="aziot-edge"
	elif [[ $package == aziot-identity-service* ]];then
		packageName="aziot-identity-service"
	elif [[ $package == lfedge-eve-tools* ]];then
		packageName="lfedge-eve-tools"
	fi
	downloadInstall $package $packageName
	if [ ${?} -ne 0 ];
	then
		echo "return value is $? packge name $package"
		echo "${package} installation failed"
		exit 2
	fi
	linkToRoot $packageName
done

postIdentityService
postEdgeConfig
