"""
    ztool is a conversion tool for docker compose to Zededa deployment template
"""

import json
import sys
import subprocess
import base64
import yaml
from docopt import docopt

usage = '''
Tool to create zedCloud edge-app from docker-compose-resource.yaml
Usage:
    ztool.py convert <docker-compose-resource.yaml>
    ztool.py convertToApp --pod-definition=<deployment.yaml> [--service-definition=<service.yaml>] [--configmap-definition=<configmap.yaml>]

'''

resources = [
    {
        "name": "resourceType",
        "value": "Tiny"
    },
    {
        "name": "cpus",
        "value": "1"
    },
    {
        "name": "memory",
        "value": "524288.00"
    }
]

appPayload = {
    "acKind": "PodManifest",
    "acVersion": "1.2.0",
    "name": "",
    "displayName": "",
    "description": "",
    "owner": {
        "user": "solution-test",
        "group": "",
        "company": "",
        "website": "www.zededa.com",
        "email": "solutions@zededa.com"
    },
    "desc": {
        "category": "",
        "os": "",
        "appCategory": "APP_CATEGORY_EDGE_APPLICATION",
        "logo": None,
        "screenshotList": {},
        "licenseList": {},
        "support": "",
        "agreementList": {}
    },
    "images": [],
    "interfaces": [],
    "vmmode": "HV_PV",
    "enablevnc": True,
    "resources": [],
    "configuration": {},
    "appType": "APP_TYPE_CONTAINER",
    "deploymentType": "DEPLOYMENT_TYPE_STAND_ALONE"
}


def cpu_unit(cpu):
    """
    method to calculate CPU Unit in deployment file
    """
    config_cpu = ""
    if cpu.endswith('m'):
        total_cpu = int(cpu.rstrip('m'))
    else:
        config_cpu = cpu

    if total_cpu < 1000:
        config_cpu = str(1)
    elif 1000 <= total_cpu <= 2000:
        config_cpu = str(2)
    elif 2000 <= total_cpu <= 3000:
        config_cpu = str(3)
    elif 3000 <= total_cpu <= 4000:
        config_cpu = str(4)
    elif 4000 <= total_cpu <= 5000:
        config_cpu = str(5)
    elif 5000 <= total_cpu <= 6000:
        config_cpu = str(6)

    return config_cpu


def convert_to_yaml(deployment_file):
    """
    Method to convert deployment template to YAML object
    """
    try:
        with open(deployment_file, encoding="utf-8") as f_input:
            data = yaml.safe_load(f_input)
    except Exception as read_error:
        print(f"read yaml {deployment_file} failed {read_error}")
    return data


def _build_custom_config(env_list):

    template_str = ""
    delimiter = "###"
    data = {
        "customConfig": {
            "name": "Custom_config",
            "add": True,
            "override": False,
            "allowStorageResize": False,
            "fieldDelimiter": delimiter,
            "template": "",
            "variableGroups": [
                {
                    "name": "Default grp 1",
                    "variables": [
                    ],
                    "required": True,
                    "condition": None
                }
            ]
        }
    }
    for item in env_list:
        config_data = {}
        temp_string = f"{item['name']}={delimiter}{item['name']}{delimiter}\n"
        template_str = template_str + temp_string
        config_data['name'] = item['name']
        config_data['label'] = item['name']
        config_data['required'] = True
        config_data['default'] = item['value']
        config_data['value'] = ""
        config_data['maxLength'] = ""
        config_data['type'] = ""
        config_data['options'] = []
        config_data['encode'] = "FILE_ENCODING_UNSPECIFIED"

        if "\\" in item['value']:
            config_data['format'] = "VARIABLE_FORMAT_FILE"
            config_data['encode'] = "FILE_ENCODING_BASE64"
        elif 'password' in item['name']:
            config_data['format'] = "VARIABLE_FORMAT_PASSWORD"
        else:
            config_data['format'] = "VARIABLE_FORMAT_TEXT"

        data['customConfig']['variableGroups'][0]['variables'].append(config_data)

    data['customConfig']['template'] = base64.b64encode(template_str.encode()).decode()
    return data


def _build_image(image):

    images = []
    image_detail = image.split('/')
    image_name = image_detail[-1]
    if isinstance(image, str):
        data = {
            'imagename': image_name,
            'maxsize': "0",
            'preserve': False,
            "target": "",
            "drvtype": "",
            "readonly": False,
            "volumelabel": "",
            "ignorepurge": True,
            "cleartext": False,
            "mountpath": ""
        }
        images.append(data)
    return images


def _build_resources(resource_data):

    sys_resources = [
        {
            "name": "resourceType",
            "value": "Custom"
        }
    ]
    if resource_data['limits']:
        container_resource = resource_data['limits']
    else:
        container_resource = resource_data['requests']

    if container_resource['cpu']:
        cpu_data = {
            'name': 'cpus',
            'value': cpu_unit(container_resource['cpu'])
        }
        sys_resources.append(cpu_data)

    if container_resource['memory']:
        memory_data = {
            'name': 'memory',
            'value': str(int(container_resource['memory']) / 1024.0)
        }
        sys_resources.append(memory_data)

    return sys_resources


def _build_outbound():

    rules = ['0.0.0.0/0', '*']
    out_bound = []
    for item in rules:
        interface = {'matches': [], 'actions': [], 'name': ""}
        if item == '*':
            match_data = {
                "type": "host",
                "value": ""
            }
        else:
            match_data = {
                "type": "ip",
                "value": item
            }
        interface['matches'].append(match_data)
        out_bound.append(interface)
    return out_bound


def _build_interfaces(ports):

    interfaces = {'name': "eth0", 'directattach': False, 'acls': []}
    for entry in _build_outbound():
        interfaces['acls'].append(entry)
    protocol = {
        "type": "protocol",
        "value": "tcp"
    }
    instance_port = {
        "type": "lport",
        "value": ""
    }
    inbound_acl = {
        "type": "ip",
        "value": "0.0.0.0/0"
    }
    for item in ports:
        inbound = {'matches': [], 'actions': [], 'name': ""}
        action_data = {
            "portmap": True,
            "portmapto": {
                "appPort": ""
            }
        }
        if 'protocol' in item.keys():
            protocol['value'] = item['protocol']

        inbound['matches'].append(protocol)
        instance_port['value'] = str(item['port'])
        inbound['matches'].append(instance_port)
        inbound['matches'].append(inbound_acl)
        action_data['portmapto']['appPort'] = item['targetPort']
        inbound['actions'].append(action_data)
        interfaces['acls'].append(inbound)

    return interfaces


def _build_volumes(volumes):

    for vol in volumes:
        data = {
            'imagename': '',
            'maxsize': "0",
            'preserve': False,
            "target": "",
            "drvtype": "",
            "readonly": False,
            "volumelabel": vol['name'],
            "ignorepurge": True,
            "cleartext": False,
            "mountpath": vol['mountPath']
        }
        appPayload['images'].append(data)

def _build_configmap(env_configmap):

    configmap_list = []
    for key, value in env_configmap.items():
        temp = {'name': key, 'value': value}
        configmap_list.append(temp)

    return _build_custom_config(configmap_list)

def convertToApp(deployment_data, service_template, configmap_template):
    """
    Method to convert pod definition to zededa container instance definition
    """

    container_data = deployment_data['spec']['template']['spec']['containers'][0]
    if 'spec' in service_template.keys():
        service_data = service_template['spec']['ports']
    else:
        service_data = {}

    if 'data' in configmap_template.keys():
        configmap_data = configmap_template['data']
    else:
        configmap_data = {}

    appPayload['name'] = container_data['name']
    if 'env' in container_data.keys() and configmap_data == {}:
        appPayload['configuration'] = _build_custom_config(container_data['env'])
    if 'env' in container_data.keys() and configmap_data != {}:
        appPayload['configuration'] = _build_configmap(configmap_data)
    if 'image' in container_data.keys():
        appPayload['images'] = _build_image(container_data['image'])
    if 'resources' in container_data.keys() and container_data['resources'] != {}:
        appPayload['resources'] = _build_resources(container_data['resources'])
    else:
        appPayload['resources'] = resources
    if 'ports' in container_data.keys() and service_data != {}:
        appPayload['interfaces'].append(_build_interfaces(service_data))
    if 'volumeMounts' in container_data.keys():
        _build_volumes(container_data['volumeMounts'])


def convertToDeployment(**kwargs):
    """
    Method convert docker compose to kubernetes pod definition
    """
    compose_file = kwargs['<docker-compose-resource.yaml>']
    cmd = f"kompose --file {compose_file} convert"
    with subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE) as proc:
        output, error = proc.communicate()

    if error:
        print(f"convert docker-compose kubectl deployment failed {error}")
        return 1
    return 0


def main():
    """
    Main Method
    """
    try:
        args = docopt(usage)
    except Exception as docopt_error:
        print(docopt_error)

    if args['convert']:
        status = convertToDeployment(**args)
        if status != 0:
            print("Conversion failed")
            sys.exit(1)
    elif args['convertToApp']:
        deployment_data = convert_to_yaml(args['--pod-definition'])
        if args['--service-definition']:
            service_data = convert_to_yaml(args['--service-definition'])
        else:
            service_data = {}
        if args['--configmap-definition']:
            config_map = convert_to_yaml(args['--configmap-definition'])
        else:
            config_map = {}

        convertToApp(deployment_data, service_data, config_map)
        print(json.dumps(appPayload, indent=3))


if __name__ == '__main__':
    main()
