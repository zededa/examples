variable "virtual_machine_name" {
    type    = list(string)
    default = [ 
        "k3s-2",
        "k3s-3" ]
}

variable "edge_node_id" {
    type    = list(string)
    default = [ 
        "d2e41884-5720-4966-81ff-b6c48bad5762",
        "c31d628a-2386-46f2-85da-0898c8cd4c8b" ]
}

variable "network_instance_name" {
    type    = list(string)
    default = [ 
        "SJC-HELIX-2-K3S-NET",
        "SJC-HELIX-3-K3S-NET" ]
}

variable "cinit_name" {
    type    = list(string)
    default = [ 
        "./c-init/k3s_2_sjc.txt",
        "./c-init/k3s_3_sjc.txt" ]
}


