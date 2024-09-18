#########Create a project with Edgeview Policy SJC ################################################################################
resource "zedcloud_project" "tech_sjc_tech_1" {
  name            = "TECH-FIELD-2024-PROJECT"
  title           = "TECH-FIELD-2024-PROJECT"
  type            = "TAG_TYPE_PROJECT"
  edgeview_policy {
      type          = "POLICY_TYPE_EDGEVIEW"
    edgeview_policy {
      access_allow_change = true
      edgeview_allow = true
      edgeviewcfg {
        app_policy {
          allow_app = true
        }
        dev_policy {
          allow_dev = true
        }
        jwt_info {
          disp_url = "zedcloud.zededa.net/api/v1/edge-view"
          allow_sec = 18000
          num_inst = 1
          encrypt = true
        }
        ext_policy {
          allow_ext = true
        }
      }
      max_expire_sec = 2592000
      max_inst = 3
    }
  }
}

######Create a datastore SJC
resource "zedcloud_datastore" "sjc_tech_ds" {
  ds_fqdn = "http://172.16.8.129"
  ds_type = "DATASTORE_TYPE_HTTP"
  name    = "TECH-FIELD-DAY-DS"
  title   = "TECH-FIELD-DAY-DS"
  ds_path = "iso"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
}

####### create FORTIGATE Image SJC #################################################################################################
resource "zedcloud_image" "sjc_fortigate_image" {
  datastore_id = zedcloud_datastore.sjc_tech_ds.id
  image_type = "IMAGE_TYPE_APPLICATION"
  image_arch = "AMD64"
  image_format = "QCOW2"
  image_sha256 = "0e275df6f35b3139d4988afcf4ddd0e3cc9fcf88320877efe0dfd17febe75147"
  image_size_bytes =  100728832
  name = "fortios-7.4.3.qcow2"
  title = "fortios-7.4.3.qcow2"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
  image_rel_url = "fortios-7.4.3.qcow2"

  depends_on = [ zedcloud_datastore.sjc_tech_ds]                       ###### I added a dependency to this image on the datastore line 36
}

#################### Create Fortigate ISO ###########################################################################################
resource "zedcloud_image" "sjc_fgt_bootstrap_iso" {       ###### This iso is used to bootstrap Fortigate config/lic
  datastore_id = zedcloud_datastore.sjc_tech_ds.id
  image_arch = "AMD64"
  image_format = "RAW"                                                  ###### RAW because its an ISO
  name = "sjc-fgt-bootstrap.iso"
  title = "sjc-fgt-bootstrap.iso"
  image_rel_url = "sjc-fgt-bootstrap.iso"
  image_sha256 = "0e8adab886b5319efa47766cd3971d38f718d4794b50cd229129d5334aef40e4"
  image_size_bytes = 380928
  image_type = "IMAGE_TYPE_APPLICATION"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
  
  depends_on = [ zedcloud_datastore.sjc_tech_ds ]
}

#################create Ubuntu Image SJC
resource "zedcloud_image" "sjc_ubuntu_image" {
  datastore_id = zedcloud_datastore.sjc_tech_ds.id
  image_type = "IMAGE_TYPE_APPLICATION"
  image_arch = "AMD64"
  image_format = "QCOW2"
  image_sha256 = "ffafb396efb0f01b2c0e8565635d9f12e51a43c51c6445fd0f84ad95c7f74f9b"
  image_size_bytes =  586285056
  name = "Ubuntu_24"
  title = "Ubuntu_24"
  project_access_list = [zedcloud_project.tech_sjc_tech_1.id]
  image_rel_url = "sjc-noble-server-cloudimg-amd64.img"

  depends_on = [ zedcloud_datastore.sjc_tech_ds]
}

######################### EDGE NODE MANAGEMENT NETWORK - How EVE physically mapped port will behave (This case set to DHCP) ###########
resource "zedcloud_network" "sjc_eve_net_port" { 
 name = "TFD-2024-EVE-NET"
 title = "TFD-2024-EVE-NET"
 description = "This Network - tells the EVE management port how to behave (DHCP...)"
 enterprise_default = false
 kind = "NETWORK_KIND_V4"
 ip {
 dhcp = "NETWORK_DHCP_TYPE_CLIENT"
 }
    project_id = zedcloud_project.tech_sjc_tech_1.id
    depends_on = [ zedcloud_project.tech_sjc_tech_1]
}




