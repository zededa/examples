################################# Deploy Rancher VM instance################################
resource "zedcloud_application_instance" "deploy_rancher_node_1" {
  name              = "rancher_instance"
  title             = "rancher_instance"
  project_id        = zedcloud_project.tech_sjc_tech_1.id
  app_id            = zedcloud_application.sjb_rancher_app.id ##### Points at the image created above in line 101
  activate          = true
  custom_config {
    add             = true
    allow_storage_resize = true
    field_delimiter = "###"
    name            = "cloud-config"
    override        = true
    template        = base64encode(file("./c-init/rancher_1_sjc.txt"))
  }
  device_id         = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
  drives {
    imagename       = zedcloud_image.sjc_ubuntu_image.name
    cleartext       = false
    ignorepurge     = true
    maxsize         = 100000000
    preserve        = false
    target          = "Disk"
    drvtype         = "HDD"
    readonly        = false
  }
  drives {
    imagename       = ""
    volumelabel     = zedcloud_volume_instance.rancher_data_persist_1.label
    drvtype         = "HDD"
    cleartext       = false
    ignorepurge     = true
    maxsize         = 50000000
    preserve        = true
    target          = "Disk"
    readonly        = false
  }
  interfaces {
    intfname = "eth0"
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    ipaddr = ""
    macaddr = ""
    netinstname = zedcloud_network_instance.sjc_lab_mgt_net.name
    privateip = false
  }
lifecycle {
  prevent_destroy = true
  }
}

########(1)#################################################################################
resource "zedcloud_application_instance" "deploy_node_3" {
  name              = "Ubuntu-Instance-1"
  title             = "Ubuntu-Instance-1"
  project_id        = zedcloud_project.tech_sjc_tech_1.id
  app_id            = zedcloud_application.sjc_ubuntu_app_3.id ##### Points at the image created above in line 101
  activate          = true
  custom_config {
    add             = true
    allow_storage_resize = true
    field_delimiter = "###"
    name            = "cloud-config"
    override        = true
    template        = base64encode(file("./c-init/ubuntu-instance-1.txt"))
  }
  device_id         = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
  drives {
    imagename       = zedcloud_image.sjc_ubuntu_image.name
    cleartext       = false
    ignorepurge     = true
    maxsize         = 100000000
    preserve        = false
    target          = "Disk"
    drvtype         = "HDD"
    readonly        = false
  }
  interfaces {
    intfname = "eth0"
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    ipaddr = ""
    macaddr = ""
    netinstname = zedcloud_network_instance.sjc_lab_lan_net.name
    privateip = false
  }
}

#########(1)######################## Deploy Ubuntu VMs instance on Trust side & Untrust side of Fortigate FW ########
resource "zedcloud_application_instance" "deploy_node_2" {
  name              = "Ubuntu-Instance-2"
  title             = "Ubuntu-Instance-2"
  project_id        = zedcloud_project.tech_sjc_tech_1.id
  app_id            = zedcloud_application.sjc_ubuntu_app_2.id ##### Points at the image created above in line 101
  activate          = true
  custom_config {
    add             = true
    allow_storage_resize = true
    field_delimiter = "###"
    name            = "cloud-config"
    override        = true
    template        = base64encode(file("./c-init/ubuntu-instance-2.txt"))
  }
  device_id         = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
  drives {
    imagename       = zedcloud_image.sjc_ubuntu_image.name
    cleartext       = false
    ignorepurge     = true
    maxsize         = 100000000
    preserve        = false
    target          = "Disk"
    drvtype         = "HDD"
    readonly        = false
  }
  interfaces {
    intfname = "eth0"
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    ipaddr = ""
    macaddr = ""
    netinstname = zedcloud_network_instance.sjc_lab_wan_net.name
    privateip = false
  }
}

########(1)######### Creates Fortigate FW#####################################################
resource "zedcloud_application_instance" "deploy_forti_1" {
  name              = "FortiGate-FW"
  title             = "FortiGate-FW"
  activate          = true
  project_id        = zedcloud_project.tech_sjc_tech_1.id
  app_id            = zedcloud_application.sjc_edge_app_fortigate.id
  custom_config {
    add             = true
    allow_storage_resize = false
    field_delimiter = ""
    name            = "cloud-config"
    override        = true
  }
  logs {
    access = true
  }
  device_id         = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
  drives {
    imagename       = zedcloud_image.sjc_fortigate_image.name
    cleartext       = false
    ignorepurge     = true
    maxsize         = 40971520
    preserve        = false
    target          = "Disk"
    drvtype         = "HDD"
    readonly        = false
  }
   drives {
    imagename       = zedcloud_image.sjc_fgt_bootstrap_iso.name
    cleartext       = false
    ignorepurge     = true
    maxsize         = 1097152
    preserve        = false
    target          = "Disk"
    drvtype         = "CDROM"
    readonly        = true
  }
  interfaces { ############ Port1 
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    intfname = "eth0"
    intforder = 1
    ipaddr = ""
    macaddr = ""
    netinstname = zedcloud_network_instance.sjc_lab_mgt_net.name
    privateip = false
  }
  interfaces { ############ Port2
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    intfname = "eth1"
    intforder = 2
    ipaddr = ""
    macaddr = ""
    netinstname = zedcloud_network_instance.sjc_lab_wan_net.name
    privateip = false   
  }
 interfaces { ############ Port3
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    intfname = "eth2"
    intforder = 3
    ipaddr = ""
    macaddr = ""
    netinstname = zedcloud_network_instance.sjc_lab_lan_net.name
    privateip = false   
  }
}

#######(Last)####### Create K3S Intance########################################################
resource "zedcloud_application_instance" "deploy_k3s_1" {
  name              = "K3S"
  title             = "K3S"
  project_id        = zedcloud_project.tech_sjc_tech_1.id
  app_id            = zedcloud_application.sjc_k3s_app_1.id ##### Points at the image created above in line 101
  activate          = true
  custom_config {
    add             = true
    allow_storage_resize = true
    field_delimiter = "###"
    name            = "cloud-config"
    override        = true
    template        = base64encode(file("./c-init/k3s_1_sjc.txt"))
  }
  device_id         = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
  drives {
    imagename       = zedcloud_image.sjc_ubuntu_image.name
    cleartext       = false
    ignorepurge     = true
    maxsize         = 100000000
    preserve        = false
    target          = "Disk"
    drvtype         = "HDD"
    readonly        = false
  }
  interfaces {
    intfname = "eth0"
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    ipaddr = ""
    macaddr = ""
    netinstname = zedcloud_network_instance.sjc_lab_lan_net.name
    privateip = false
  }
}

########(Last)####### Create K3S Intance########################################################
resource "zedcloud_application_instance" "deploy_linux_node_1" {
  count             = length(var.virtual_machine_name)
  name              = var.virtual_machine_name[count.index]
  title             = var.virtual_machine_name[count.index]
  project_id        = zedcloud_project.tech_sjc_tech_1.id
  app_id            = zedcloud_application.k3s_single_node_multi.id ##### Points at the image created above in line 101
  activate          = true
  custom_config {
    add             = true
    allow_storage_resize = true
    field_delimiter = "###"
    name            = "cloud-config"
    override        = true
    template        = base64encode(file(var.cinit_name[count.index]))
  }
  device_id         = var.edge_node_id[count.index]
  drives {
    imagename       = zedcloud_image.sjc_ubuntu_image.name
    cleartext       = false
    ignorepurge     = true
    maxsize         = 20971520
    preserve        = false
    target          = "Disk"
    drvtype         = "HDD"
    readonly        = false
  }
  interfaces {
    intfname = "eth0"
    directattach = false
    access_vlan_id = 0
    default_net_instance = false
    ipaddr = ""
    macaddr = ""
    netinstname = var.network_instance_name[count.index]
    privateip = false
  }
}