usage = '''
enterprise management example
Usage:
  enterprise_manage.py login --username=<username> --password=<password>
  enterprise_manage.py user <command> <name> <type> --email=<email> [--role=<role>] [--password=<password>] [--fullname=<firstname>] [--phone=<phone>] [--timezone=<timezone>]
  enterprise_manage.py user <command> <name> [--allowed-enterprise=<allowed-enterprise>...]
  enterprise_manage.py enterprise <command> <name> --inherit-auth
  
'''
from docopt import docopt
import os, sys, json
from libs.zapi import zapi

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

def user(args):

    if args['<command>'] == 'create':
        return user_create(args)
    if args['<command>'] == 'update':
        return user_update(args)
    if args['<command>'] == 'delete':
        return user_delete(args)

def enterprise(args):

    if args['<command>'] == 'create':
        return enterprise_create(args)
    if args['<command>'] == 'update':
        return enterprise_update(args)

def user_create(args):

    print("=" * 100)
    zmethod = "user create"
    print(zmethod.center(70))
    print("=" * 100)

    payload = {}
    url_ext = "/api/v1/users"
    payload['username'] = args['<name>']
    if args['<type>'] == "Local":
        payload['type'] = "AUTH_TYPE_LOCAL"
    else:
        payload['type'] = "AUTH_TYPE_OAUTH"
    payload['email'] = args['--email']
    if args['--role']:
        payload['roleId'] = get_role_id(args['--role'])
    if args['--fullname']:
        payload['fullName'] = args['--fullname']
    else:
        payload['fullName'] = args['<name>']
    if args['--timezone']:
        payload['timeZone'] = args['--timezone']
    if args['--phone']:
        payload['phone'] = args[''--phone]
    status, response = zsession.post_request(url_ext, payload)
    if status != 0:
        sys.exit()
    if args['<type>'] == "Local":
        status = create_user_credentials(args['<name>'], args['--password'])

def user_update(args):

    print("=" * 100)
    zmethod = "user update"
    print(zmethod.center(70))
    print("=" * 100)

    url_ext = f"/api/v1/users/name/{args['<name>']}"
    status, response = zsession.get_request(url_ext)
    payload = response
    allowed_enterprises = []
    if status != 0:
        sys.exit()
    if args['--allowed-enterprise']:

        for items in args['--allowed-enterprise']:
            details = {}
            enterprise_name, role = items.split(':')
            details['id'] = get_enterprise_id(enterprise_name)
            details['name'] = enterprise_name
            details['roleId'] = get_role_id(role)
            allowed_enterprises.append(details)
    if payload['allowedEnterprises']:
        payload['allowedEnterprises'] = payload['allowedEnterprises'] + allowed_enterprises
    else:
        payload['allowedEnterprises'] = allowed_enterprises

    url_ext_user=f"/api/v1/users/id/{payload['id']}"
    status, response = zsession.put_request(url_ext_user, payload)
    if status != 0:
        sys.exit()


def user_delete():

    print("=" * 100)
    zmethod = "user Delete"
    print(zmethod.center(70))
    print("=" * 100)
    pass

def enterprise_create(args):

    print("=" * 100)
    zmethod = "Enterprise create with inheritAuthFromParent"
    print(zmethod.center(70))
    print("=" * 100)

    payload = {
        'name': args['<name>'],
        'title': args['<name>'],
        'inheritAuthFromParent': True if args['--inherit-auth'] else False
    }

    status, response = zsession.post_request('/api/v1/enterprises', payload)
    if status != 0:
        sys.exit()

def enterprise_update(args):
    pass

def get_role_id(role):

    url_ext = f"/api/v1/roles/name/{role}"
    status, response = zsession.get_request(url_ext)
    if status != 0:
        print("POST request to get ID of {role} failed")
        sys.exit()
    return response['id']

def get_enterprise_id(name):

    url_ext = f"/api/v1/enterprises/name/{name}"
    status, response = zsession.get_request(url_ext)
    if status != 0:
        sys.exit()
    return response['id']


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

    base_url = "https://zedcontrol.hummingbird.zededa.net"
    if args['login']:
        user_config = create_config(args['--username'], args['--password'])
        zsession = zapi(base_url, user_config)
        status = login()
        if status != 0:
            print("Login to the zedcontrol failed")
            sys.exit()
    else:
        user_config = read_config()
        zsession = zapi(base_url, user_config)

    if args['user']:
        status = user(args)
    if args['enterprise']:
        status = enterprise(args)


if __name__ == '__main__':
    main()


