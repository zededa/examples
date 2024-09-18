########################## Creates Network Instances #####################################
########################## Creates 3 x Network Instances on helix 1
resource "zedcloud_network_instance" "sjc_lab_mgt_net" {
  name = "SJC-MGT-NET"
  title = "SJC-MGT-NET"
  kind = "NETWORK_INSTANCE_KIND_LOCAL"
  type = "NETWORK_INSTANCE_DHCP_TYPE_V4"
  port = "eth0"
  device_id = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
  ip {
    dhcp_range {
    end = "10.20.0.30"
    start = "10.20.0.20"
  }
    dns = [
      "1.1.1.1"
  ]
    domain = ""
    gateway = "10.20.0.1"
    ntp = "64.246.132.14"
    subnet = "10.20.0.0/24"
  }
 depends_on = [ zedcloud_edgenode.edgenode_create_sjc_onlogic ]
}

resource "zedcloud_network_instance" "sjc_lab_wan_net" {
  name = "WAN-NET"
  title = "WAN-NET"
  kind = "NETWORK_INSTANCE_KIND_SWITCH"
  type = "NETWORK_INSTANCE_DHCP_TYPE_UNSPECIFIED"
  port = ""
  device_id = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
# depends_on = [  ]
}

resource "zedcloud_network_instance" "sjc_lab_lan_net" {
  name = "LAN-NET"
  title = "LAN-NET"
  kind = "NETWORK_INSTANCE_KIND_SWITCH"
  type = "NETWORK_INSTANCE_DHCP_TYPE_UNSPECIFIED"
  port = ""
  device_id = zedcloud_edgenode.edgenode_create_sjc_onlogic.id
# depends_on = [  ]
}

########################## Creates a network on SJC Helix EDGE NODE 2 #####################
resource "zedcloud_network_instance" "sjc_lab_helix_2" {
  name = "SJC-HELIX-2-K3S-NET"
  title = "SJC-HELIX-2-K3S-NET"
  kind = "NETWORK_INSTANCE_KIND_LOCAL"
  type = "NETWORK_INSTANCE_DHCP_TYPE_V4"
  port = "eth0"
  device_id = "d2e41884-5720-4966-81ff-b6c48bad5762"
  ip {
    dhcp_range {
    end = "10.30.0.30"
    start = "10.30.0.20"
  }
    dns = [
      "1.1.1.1"
  ]
    domain = ""
    gateway = "10.30.0.1"
    ntp = "64.246.132.14"
    subnet = "10.30.0.0/24"
  }
 depends_on = [ zedcloud_edgenode.edgenode_create_sjc_onlogic ]
}

########################## Creates a network on SJC Helix EDGE NODE 3 #####################
resource "zedcloud_network_instance" "sjc_lab_helix_3" {
  name = "SJC-HELIX-3-K3S-NET"
  title = "SJC-HELIX-3-K3S-NET"
  kind = "NETWORK_INSTANCE_KIND_LOCAL"
  type = "NETWORK_INSTANCE_DHCP_TYPE_V4"
  port = "eth0"
  device_id = "c31d628a-2386-46f2-85da-0898c8cd4c8b"
  ip {
    dhcp_range {
    end = "10.30.0.30"
    start = "10.30.0.20"
  }
    dns = [
      "1.1.1.1"
  ]
    domain = ""
    gateway = "10.30.0.1"
    ntp = "64.246.132.14"
    subnet = "10.30.0.0/24"
  }
 depends_on = [ zedcloud_edgenode.edgenode_create_sjc_onlogic ]
}