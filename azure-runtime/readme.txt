Build Image:
===========


	1. Install qemu-kvm: 
                 Install following command to install KVM and additional virtualization management packages 
         
                   	sudo apt-get update
                   	sudo apt-get install qemu-kvm libvirt-bin bridge-utils virtinst virt-manager
                   
	    Once the package is installed, verify libvirt daemon started automatically 
		
		sudo systemctl  is-active libvirtd
 
	   Add user to libvirt and kvm group 

                       sudo usermod -aG libvirt $USER
                       sudo usermod -aG kvm $USER

	2. Install packer:
               Browse to https://www.packer.io/downloads.html . Look fornthe linux section, right click on the 64-bit option and select and copy the link address. Go to terminal and perform below command
 
              		wget <paste the URL copied from packer.io>

               Unzip the package and move the packer to /usr/local/bin

	3. Clone the example repository 
	4. Replace lfedge-eve-tools.deb download URL in iotedge-install.sh
		A. cd azure-runtime
		B. Copy the lfedge-eve-tools.deb to the local http server 
		C. replace download URL inside scripts/iotedge-install.sh, line no: 185
	5. Build Image:
      	      	packer build Ubuntu1804-packer.json
	6. Image output dir: ubuntu-18.04-runtime-1.4/Ubuntu-18.04-runtime*

