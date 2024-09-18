####################### Onboards EDGE NODEs ########################################
resource "zedcloud_edgenode" "edgenode_create_sjc_onlogic_1" {
  model_id = "160eadc4-b85f-43e8-80df-7e9d9bd50f50"
  name = "ONLOGIC-COMPUTE-1-SJC"
  title = "ONLOGIC-COMPUTE-1-SJC"
  project_id = zedcloud_project.tech_sjc_tech_1.id
  onboarding_key = "<onboarding key>"
  serialno = "<hard serial -r soft serial number>"
  description = "Tech Field Day Demo"
  admin_state = "ADMIN_STATE_ACTIVE"
    config_item {
        bool_value   = false
        float_value  = 0
        key          = "debug.enable.ssh" ### Manny public key
        string_value = "<ssh pub key>"
        uint32_value = 0
        uint64_value = 0
    }
    edgeviewconfig {
        generation_id = 0
        token         = "<zecontrol token for using EdgeView feature>"

        app_policy {
            allow_app = true
        }

        dev_policy {
            allow_dev = true
        }

        ext_policy {
            allow_ext = true
        }

        jwt_info {
            allow_sec  = 18000
            disp_url   = "zedcloud.zededa.net/api/v1/edge-view"
            encrypt    = true
            expire_sec = "0"
            num_inst   = 3
        }  
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_MANAGEMENT"
        intfname   = "eth0"
        netname    = zedcloud_network.sjc_eve_net_port.name
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth1"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth2"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth3"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth4"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth5"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "Audio"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM1"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM2"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM3"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "USB"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "VGA"
        tags       = {}
    }
 
    depends_on = [ 
        zedcloud_network.sjc_eve_net_port]

 lifecycle {
   ignore_changes = [ 
    interfaces,
    edgeviewconfig
    ]
 }
}

resource "zedcloud_edgenode" "edgenode_create_sjc_onlogic_2" {
  model_id = "160eadc4-b85f-43e8-80df-7e9d9bd50f50"
  name = "ONLOGIC-COMPUTE-2-SJC"
  title = "ONLOGIC-COMPUTE-2-SJC"
  project_id = zedcloud_project.tech_sjc_tech_1.id
  onboarding_key = "<onboarding key>"
  serialno = "<hard serial -r soft serial number>"
  description = "Tech Field Day Demo"
  admin_state = "ADMIN_STATE_ACTIVE"
    config_item {
        bool_value   = false
        float_value  = 0
        key          = "debug.enable.ssh" ### Manny public key
        string_value = "<ssh pub key>"
        uint32_value = 0
        uint64_value = 0
    }
    edgeviewconfig {
        generation_id = 0
        token         = "<zecontrol token for using EdgeView feature>"

        app_policy {
            allow_app = true
        }

        dev_policy {
            allow_dev = true
        }

        ext_policy {
            allow_ext = true
        }

        jwt_info {
            allow_sec  = 18000
            disp_url   = "zedcloud.zededa.net/api/v1/edge-view"
            encrypt    = true
            expire_sec = "0"
            num_inst   = 3
        }  
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_MANAGEMENT"
        intfname   = "eth0"
        netname    = zedcloud_network.sjc_eve_net_port.name
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth1"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth2"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth3"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth4"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth5"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "Audio"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM1"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM2"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM3"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "USB"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "VGA"
        tags       = {}
    }
 
    depends_on = [ 
        zedcloud_network.sjc_eve_net_port]

 lifecycle {
   ignore_changes = [ 
    interfaces,
    edgeviewconfig
    ]
 }
}

resource "zedcloud_edgenode" "edgenode_create_sjc_onlogic_3" {
  model_id = "160eadc4-b85f-43e8-80df-7e9d9bd50f50"
  name = "ONLOGIC-COMPUTE-3-SJC"
  title = "ONLOGIC-COMPUTE-3-SJC"
  project_id = zedcloud_project.tech_sjc_tech_1.id
  onboarding_key = "<onboarding key>"
  serialno = "<hard serial -r soft serial number>"
  description = "Tech Field Day Demo"
  admin_state = "ADMIN_STATE_ACTIVE"
    config_item {
        bool_value   = false
        float_value  = 0
        key          = "debug.enable.ssh" ### Manny public key
        string_value = "<ssh pub key>"
        uint32_value = 0
        uint64_value = 0
    }
    edgeviewconfig {
        generation_id = 0
        token         = "<zecontrol token for using EdgeView feature>"

        app_policy {
            allow_app = true
        }

        dev_policy {
            allow_dev = true
        }

        ext_policy {
            allow_ext = true
        }

        jwt_info {
            allow_sec  = 18000
            disp_url   = "zedcloud.zededa.net/api/v1/edge-view"
            encrypt    = true
            expire_sec = "0"
            num_inst   = 3
        }  
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_MANAGEMENT"
        intfname   = "eth0"
        netname    = zedcloud_network.sjc_eve_net_port.name
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth1"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth2"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth3"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth4"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "eth5"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_SHARED"
        intfname   = "Audio"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM1"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM2"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "COM3"
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "USB"
        netname    = ""
        tags       = {}
    }
    interfaces {
        cost       = 0
        intf_usage = "ADAPTER_USAGE_APP_DIRECT"
        intfname   = "VGA"
        tags       = {}
    }
 
    depends_on = [ 
        zedcloud_network.sjc_eve_net_port]

 lifecycle {
   ignore_changes = [ 
    interfaces,
    edgeviewconfig
    ]
 }
}