usage = '''
Create Datastore, image & edge-app as global object
Supported datastore <type> are HTTP, HTTPS, AWSS3, AZUREBLOB, CONTAINERREGISTRY
Default origin type is Local

Usage:
  apiHelper.py login --username=<username> --password=<password>
  apiHelper.py datastore create <name> <type> [--fqdn=<fqdn>] [--region=<region>] [--apikey=<apikey>] [--apipass=<password>] [--dpath=<dpath>] [--origin-type=<origin-type>] [--description=<description>]
  apiHelper.py image create <name> --datastore=<datastore_name> --arch=[AMD64|ARM64] [--origin-type=<global|local>] [--description=<description>]
  apiHelper.py image uplink <name> --image-sha=<image-sha> --image-size=<image-size>
  apiHelper.py edge-app create <name> --manifest=<manifest.json> [--description=<description>] [--origin-type=<global|local>]
  apiHelper.py datastore update <name>
  apiHelper.py image update <name>
  apiHelper.py edge-app update <name>
  apiHelper.py edge-app refresh <name>
'''

from docopt import docopt
import os, sys, json
from libs.zapi import zapi
from libs.resourceCreate import *

def login():

    print("=" * 100)
    zmethod = "login"
    print(zmethod.center(70))
    print("=" * 100)
    username = zsession.username
    password = zsession.password
    status, token = zsession.login()
    if status != 0:
        print(f"login failed reason: {token}")
        return 1
    user_config = {
        'username': zsession.username,
        'password': zsession.password,
        'auth_token': token
    }

    with open('config.json', 'w') as f:
        f.write(json.dumps(user_config, indent=4))
    return 0


def create_user_credentials(username, password):

    payload = {
        'owner': username,
        'type': 'CREDENTIAL_TYPE_PASSWORD',
        'newCred': password
    }
    status, response = zsession.post_request('/api/v1/credentials', payload)
    return status

def create_config(username, password):

    user_data = {
        "username": username,
        "password": password
    }
    with open('config.json', 'w') as f:
        f.write(json.dumps(user_data, indent=4))
    return user_data

def read_config():
    file_name = "config.json"
    with open("config.json", 'r') as f:
        user_config = json.load(f)
    return user_config

def main():

    global zsession
    try:
        args = docopt(usage)
    except Exception as e:
        print(e)

    base_url = "https://zedcontrol.canary.zededa.net"
    if args['login']:
        user_config = create_config(args['--username'], args['--password'])
        zsession = zapi(base_url, user_config)
        status = login()
        if status != 0:
            print("Login to the zedCloud failed")
            sys.exit()
    else:
        user_config = read_config()
        zsession = zapi(base_url, user_config)

    if args['datastore'] and args['create']:
        urlExt = "/api/v1/datastores"
        status, payload = constructDs(args)
        if status != 0:
            print("Construct payload for Datastore failed")
            sys.exit(1)
        status, response = zsession.post_request(urlExt, payload)
    elif args['datastore'] and args['update']:
        status = updateDataStore(zsession, args)

    if args['image'] and args['create']:
        url = f"/api/v1/datastores/name/{args['--datastore']}"
        status, response = zsession.get_request(url)
        if status != 0:
            print(f"Get datastore {args['<name>']} failed")
            sys.exit(1)
        status, payload = imageCreate(args, response['id'])
        url_image= "/api/v1/apps/images"
        status, response = zsession.post_request(url_image, payload)
    elif args['image'] and args['uplink']:
        uplinkStatus = imageUplink(zsession, args)
    elif args['image'] and args['update']:
        updateStatus = updateImage(zsession, args)

    if args['edge-app'] and args['create']:
        status = edgeAppCreate(zsession, args)
    elif args['edge-app'] and args['update']:
        updateStatus = updateEdgeApp(zsession, args)
    elif args['edge-app'] and args['refresh']:
        status = edgeAppRefresh(zsession, args)


if __name__ == '__main__':
    main()
