#!/bin/bash

postConfig() {

	#Create docker group
	containerdService="/zed/data1/moby-containerd/lib/systemd/system/containerd.service"
    dockerService="/zed/data1/moby-engine/lib/systemd/system/docker.service"
	dockerSocket="/zed/data1/moby-engine/lib/systemd/system/docker.socket"
	multiUserService="/etc/systemd/system/multi-user.target.wants/docker.service"
	dockerServiceDst="/etc/systemd/system/sockets.target.wants/docker.socket"
	containerdServiceDst="/etc/systemd/system/multi-user.target.wants/containerd.service"
	containerdataRoot="/zed/data3/docker"

	#groupadd -g 113 docker
	grep  -i "^docker:" /etc/group
	if [ $? -ne 0 ];
	then
		echo "group name docker not present"
		groupadd -g 113 docker
	fi

	rm -rf /lib/systemd/system/containerd.service
	cp -rf $containerdService /lib/systemd/system/containerd.service
	ln -s /lib/systemd/system/containerd.service $containerdServiceDst
	rm -rf /lib/systemd/system/docker.service
	cp -rf $dockerService /lib/systemd/system/docker.service
	ln -s /lib/systemd/system/docker.service $multiUserService
	rm -rf /lib/systemd/system/docker.socket
	cp -rf $dockerSocket /lib/systemd/system/docker.socket
	ln -s /lib/systemd/system/docker.socket $dockerServiceDst
	cp -rf /home/pocuser/daemon.json /etc/docker/

	if [ ! -d $containerdataRoot ]; then
		mkdir -p $containerdataRoot
	fi

	systemctl enable docker || true
	systemctl start docker || true
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

	case "$1" in 
		pigz)
			pigzPackage="/zed/data1/pigz"
			walkDir $pigzPackage $pigzPackage
			;;
		moby-buildx)
			mobyBuildxPackage="/zed/data1/moby-buildx"
			walkDir $mobyBuildxPackage $mobyBuildxPackage
			;;
		moby-containerd)
			mobyContainerPackage="/zed/data1/moby-containerd"
			walkDir $mobyContainerPackage $mobyContainerPackage
			;;
		moby-runc)
			mobyRuncPackage="/zed/data1/moby-runc"
			walkDir $mobyRuncPackage $mobyRuncPackage
			;;
		moby-cli)
			mobyCliPackage="/zed/data1/moby-cli"
			walkDir $mobyCliPackage $mobyCliPackage
			;;
		moby-engine)
			mobyEnginePackage="/zed/data1/moby-engine"
			walkDir $mobyEnginePackage $mobyEnginePackage
			;;
		*)
			echo "not a valid option"
	esac
}

downloadInstall() {
	packageName=$1
	packageLocation="/zed/data1/$1"
	if [ ! -d $packageLocation ]; then
		mkdir -p $packageLocation
	fi
    cd $packageLocation
    apt download $packageName
    if [ ${?} -ne 0 ];
	then
		echo "Download ${packageName} failed"
		exit 2
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
packageList=("pigz" "moby-buildx" "moby-containerd" "moby-runc" "moby-cli" "moby-engine")

for package in ${packageList[@]}
do
	echo "$package"
    	downloadInstall $package
	if [ ${?} -ne 0 ];
	then
		echo "return value $? package Name $package"
		echo "${package} installation failed"
		exit 2
	fi
	linkToRoot $package
	if [ $? -ne 0 ];
	then
		echo "Installing and linking extracted file return non zero value"
		exit 2
	fi
done
postConfig

