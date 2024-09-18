################################ Rancher App #####################################################################
resource "zedcloud_application" "sjc_rancher_app" {
  name = "rancher_app"
  title = "rancher_app"
  networks = 1
  cpus = 2
  memory = 4000000
  storage = 100000000
  manifest {
    ac_kind = "VMManifest"
    ac_version = "1.2.0"
    name = "ubuntu_24"
  owner {
    user = "Manny"
    company = "Zededa"
    website = "www.zededa.com"
    email = "manny@zededa.com"
  }
  desc {
    app_category = "APP_CATEGORY_UNSPECIFIED"
    category = "APP_CATEGORY_OPERATING_SYSTEM"
    } 
  images {
    imagename = zedcloud_image.sjc_ubuntu_image.name
    imageid = zedcloud_image.sjc_ubuntu_image.id
    imageformat = "QCOW2"
    cleartext = false
    drvtype = "HDD"
    ignorepurge = true
    maxsize = 100000000
    target = "Disk"
    }
  images {
    volumelabel = zedcloud_volume_instance.rancher_data_persist_1.label
    imageformat = "QCOW2"
    cleartext = false
    drvtype = "HDD"
    ignorepurge = true
    maxsize = 2500000
    target = "Disk"
    }
  interfaces {
    name = "eth0"
    type = ""
    directattach = false
    privateip = false
   acls {
      matches { ### Outbound rule 
        type = "ip"
        value = "0.0.0.0/0"
      }
    }
    acls {
      matches { ### Outbound rule 
        type = "ip"
        value = "0.0.0.0/0"
      }
    }
    acls { 
      matches { ### Inbound rules & port mappings
        type = "ip"
        value = "0.0.0.0/0"
      }
      actions {
        portmap = true
        portmapto {
          app_port = 22 #Internal App port
        }
      }
      matches {
        type = "protocol"
        value = "tcp"
      }
      matches {
        type = "lport"
        value = 1022  ### External Edge node port
     }
      matches {
        type = "ip"
        value = "0.0.0.0/0"
     }
    }
    acls { 
      matches { ### Mapping inbound port 10443-443
        type = "ip"
        value = "0.0.0.0/0"
      }
      actions {
        portmap = true
        portmapto {
          app_port = 10443 #Internal App port
        }
      }
      matches {
        type = "protocol"
        value = "tcp"
      }
      matches {
        type = "lport"
        value = 20443 #External Edge node port
     }
      matches {
        type = "ip"
        value = "0.0.0.0/0"
     }
    }
   }
 
  vmmode = "HV_HVM"
  enablevnc = true

  resources {
    name = "resourceType"
    value = "custom"
  }
  resources {
    name = "cpus"
    value = 2
  }
  resources {
    name = "memory"
    value = 4000000
  }
  resources {
    name = "storage"
    value = 100000000
  }
  configuration {
    custom_config {
      add = true
      name = "cloud-config"
      override = true
      template = ""      
    }
   }
  app_type = "APP_TYPE_VM" 
  deployment_type = "DEPLOYMENT_TYPE_STAND_ALONE"
  cpu_pinning_enabled = false
 }
  user_defined_version = "24.04"
  origin_type = "ORIGIN_LOCAL"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
}

############Creates automated end to end VM -> FW -> (allow policy) -> VM#########################################
#######################################  VM -> FW -> (allow policy + NAT) -> Internet#############################
####################################### Creates Forti App#########################################################
resource "zedcloud_application" "sjc_edge_app_fortigate" {
  name = "fortigate-7.4.3"
  title = "Fortigate-7.4.3"
  networks = 3
  cpus = 4
  memory = 8000000
  storage = 40971520
  manifest {
    ac_kind = "VMManifest"
    ac_version = "1.2.0"
    name = "Fortigate-7.4.3"
  owner {
    user = "Manny Calero"
    company = "Zededa"
    website = "www.zededa.com"
    email = "manny@zededa.com"
  }
  desc {
    app_category = "APP_CATEGORY_UNSPECIFIED"
    category = "APP_CATEGORY_SECURITY"
    } 
  images {
    imagename = zedcloud_image.sjc_fortigate_image.name
    imageid = zedcloud_image.sjc_fortigate_image.id
    imageformat = "QCOW2"
    cleartext = false
    drvtype = "HDD"
    ignorepurge = true
    maxsize = 40971520
    target = "Disk"
 
    }
  images {
    imagename = zedcloud_image.sjc_fgt_bootstrap_iso.name
    imageid = zedcloud_image.sjc_fgt_bootstrap_iso.id
    imageformat = "RAW"
    cleartext = false
    drvtype = "CDROM"
    ignorepurge = true
    maxsize = 1097152
    target = "Disk"
    }
  interfaces {
    name = "eth0"
    type = ""
    directattach = false
    privateip = false
    acls {
      matches { ### Outbound rule 
        type = "ip"
        value = "0.0.0.0/0"
      }
    }
    acls { 
      matches { ### Inbound rules & port mappings
        type = "ip"
        value = "0.0.0.0/0"
      }
      actions {
        portmap = true
        portmapto {
          app_port = 22 #Internal App port
        }
      }
      matches {
        type = "protocol"
        value = "tcp"
      }
      matches {
        type = "lport"
        value = 11122  ### External Edge node port
     }
      matches {
        type = "ip"
        value = "0.0.0.0/0"
     }
    }
        acls { 
      matches { ### Mapping inbound port 10443-443
        type = "ip"
        value = "0.0.0.0/0"
      }
      actions {
        portmap = true
        portmapto {
          app_port = 443 #Internal App port
        }
      }
      matches {
        type = "protocol"
        value = "tcp"
      }
      matches {
        type = "lport"
        value = 10443 #External Edge node port
     }
      matches {
        type = "ip"
        value = "0.0.0.0/0"
     }
    }
   } 
  interfaces {
    name = "eth1"
    type = ""
    directattach = false
    privateip = false
      acls {
        matches { ### Outbound rule
          type = "ip"
          value = "0.0.0.0/0"
      }
    }
   }
  interfaces {
    name = "eth2"
    type = ""
    directattach = false
    privateip = false
      acls {
        matches { ### Outbound rule
          type = "ip"
          value = "0.0.0.0/0"
        }
      }
   }
  vmmode = "HV_HVM"
  enablevnc = true

  resources {
    name = "resourceType"
    value = "custom"
  }
  resources {
    name = "cpus"
    value = 4
  }
  resources {
    name = "memory"
    value = 8000000
  }
  resources {
    name = "storage"
    value = 40971520
  }
  configuration {
    custom_config {
      add = true
      name = "cloud-config"
      override = true
      template = ""      
    }
   }
  app_type = "APP_TYPE_VM" 
  deployment_type = "DEPLOYMENT_TYPE_STAND_ALONE"
  cpu_pinning_enabled = false
 }
  user_defined_version = "7.4.3"
  origin_type = "ORIGIN_LOCAL"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
}

############################### Creates Ubuntu Application #######################################################
resource "zedcloud_application" "sjc_k3s_app_1" {
  name = "k3s_app_1"
  title = "k3s_app_1"
  networks = 1
  cpus = 4
  memory = 8000000
  storage = 25000000
  manifest {
    ac_kind = "VMManifest"
    ac_version = "1.2.0"
    name = "ubuntu_24"
  owner {
    user = "Manny"
    company = "Zededa"
    website = "www.zededa.com"
    email = "manny@zededa.com"
  }
  desc {
    app_category = "APP_CATEGORY_UNSPECIFIED"
    category = "APP_CATEGORY_NETWORKING"
    } 
  images {
    imagename = zedcloud_image.sjc_ubuntu_image.name
    imageid = zedcloud_image.sjc_ubuntu_image.id
    imageformat = "QCOW2"
    cleartext = false
    drvtype = "HDD"
    ignorepurge = true
    maxsize = 25000000
    target = "Disk"
 
    }
  interfaces {
    name = "eth0"
    type = ""
    directattach = false
    privateip = false
     acls { 
      matches { ### Inbound rules & port mappings
        type = "ip"
        value = "0.0.0.0/0"
      }
    }
  }
  vmmode = "HV_HVM"
  enablevnc = true

  resources {
    name = "resourceType"
    value = "custom"
  }
  resources {
    name = "cpus"
    value = 4
  }
  resources {
    name = "memory"
    value = 8000000
  }
  resources {
    name = "storage"
    value = 25000000
  }
  configuration {
    custom_config {
      add = true
      name = "cloud-config"
      override = true
      template = ""      
    }
   }
  app_type = "APP_TYPE_VM" 
  deployment_type = "DEPLOYMENT_TYPE_STAND_ALONE"
  cpu_pinning_enabled = false
 }
  user_defined_version = "24.04"
  origin_type = "ORIGIN_LOCAL"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
}

resource "zedcloud_application" "sjc_ubuntu_app_2" {
  name = "ubuntu_app_2"
  title = "ubuntu_app_2"
  networks = 1
  cpus = 2
  memory = 4000000
  storage = 25000000
  manifest {
    ac_kind = "VMManifest"
    ac_version = "1.2.0"
    name = "ubuntu_24"
  owner {
    user = "Manny"
    company = "Zededa"
    website = "www.zededa.com"
    email = "manny@zededa.com"
  }
  desc {
    app_category = "APP_CATEGORY_UNSPECIFIED"
    category = "APP_CATEGORY_NETWORKING"
    } 
  images {
    imagename = zedcloud_image.sjc_ubuntu_image.name
    imageid = zedcloud_image.sjc_ubuntu_image.id
    imageformat = "QCOW2"
    cleartext = false
    drvtype = "HDD"
    ignorepurge = true
    maxsize = 25000000
    target = "Disk"
 
    }
  interfaces {
    name = "eth0"
    type = ""
    directattach = false
    privateip = false
     acls { 
      matches { ### Inbound rules & port mappings
        type = "ip"
        value = "0.0.0.0/0"
      }
    }
  }
  vmmode = "HV_HVM"
  enablevnc = true

  resources {
    name = "resourceType"
    value = "custom"
  }
  resources {
    name = "cpus"
    value = 4
  }
  resources {
    name = "memory"
    value = 4000000
  }
  resources {
    name = "storage"
    value = 25000000
  }
  configuration {
    custom_config {
      add = true
      name = "cloud-config"
      override = true
      template = ""      
    }
   }
  app_type = "APP_TYPE_VM" 
  deployment_type = "DEPLOYMENT_TYPE_STAND_ALONE"
  cpu_pinning_enabled = false
 }
  user_defined_version = "24.04"
  origin_type = "ORIGIN_LOCAL"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
}

resource "zedcloud_application" "sjc_ubuntu_app_3" {
  name = "ubuntu_app_3"
  title = "ubuntu_app_3"
  networks = 1
  cpus = 2
  memory = 4000000
  storage = 25000000
  manifest {
    ac_kind = "VMManifest"
    ac_version = "1.2.0"
    name = "ubuntu_24"
  owner {
    user = "Manny"
    company = "Zededa"
    website = "www.zededa.com"
    email = "manny@zededa.com"
  }
  desc {
    app_category = "APP_CATEGORY_UNSPECIFIED"
    category = "APP_CATEGORY_NETWORKING"
    } 
  images {
    imagename = zedcloud_image.sjc_ubuntu_image.name
    imageid = zedcloud_image.sjc_ubuntu_image.id
    imageformat = "QCOW2"
    cleartext = false
    drvtype = "HDD"
    ignorepurge = true
    maxsize = 25000000
    target = "Disk"
 
    }
  interfaces {
    name = "eth0"
    type = ""
    directattach = false
    privateip = false
     acls { 
      matches { ### Inbound rules & port mappings
        type = "ip"
        value = "0.0.0.0/0"
      }
    }
  }
  vmmode = "HV_HVM"
  enablevnc = true

  resources {
    name = "resourceType"
    value = "custom"
  }
  resources {
    name = "cpus"
    value = 4
  }
  resources {
    name = "memory"
    value = 4000000
  }
  resources {
    name = "storage"
    value = 25000000
  }
  configuration {
    custom_config {
      add = true
      name = "cloud-config"
      override = true
      template = ""      
    }
   }
  app_type = "APP_TYPE_VM" 
  deployment_type = "DEPLOYMENT_TYPE_STAND_ALONE"
  cpu_pinning_enabled = false
 }
  user_defined_version = "24.04"
  origin_type = "ORIGIN_LOCAL"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
}

############## Used for multiple K3S instances ####################################################################
resource "zedcloud_application" "k3s_single_node_multi" {
  name = "k3s_single_node"
  title = "k3s_single_node"
  networks = 1
  cpus = 2
  memory = 4000000
  storage = 25000000
  manifest {
    ac_kind = "VMManifest"
    ac_version = "1.2.0"
    name = "ubuntu_24"
  owner {
    user = "Manny"
    company = "Zededa"
    website = "www.zededa.com"
    email = "manny@zededa.com"
  }
  desc {
    app_category = "APP_CATEGORY_UNSPECIFIED"
    category = "APP_CATEGORY_NETWORKING"
    } 
  images {
    imagename = zedcloud_image.sjc_ubuntu_image.name
    imageid = zedcloud_image.sjc_ubuntu_image.id
    imageformat = "QCOW2"
    cleartext = false
    drvtype = "HDD"
    ignorepurge = true
    maxsize = 25000000
    target = "Disk"
 
    }
  interfaces {
    name = "eth0"
    type = ""
    directattach = false
    privateip = false
     acls { 
      matches { ### Inbound rules & port mappings
        type = "ip"
        value = "0.0.0.0/0"
      }
    }
  }
  vmmode = "HV_HVM"
  enablevnc = true

  resources {
    name = "resourceType"
    value = "custom"
  }
  resources {
    name = "cpus"
    value = 2
  }
  resources {
    name = "memory"
    value = 4000000
  }
  resources {
    name = "storage"
    value = 25000000
  }
  configuration {
    custom_config {
      add = true
      name = "cloud-config"
      override = true
      template = ""      
    }
   }
  app_type = "APP_TYPE_VM" 
  deployment_type = "DEPLOYMENT_TYPE_STAND_ALONE"
  cpu_pinning_enabled = false
 }
  user_defined_version = "24.04"
  origin_type = "ORIGIN_LOCAL"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
}
