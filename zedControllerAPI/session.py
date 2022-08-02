import os, sys, json
from libs.zapi import zapi


def login():
    status, response = zsession.login()
    print(f"{status}")

def getSession():
    url_ext = "/api/v1/sessions"
    return zsession.get_request(url_ext)


def refreshSession():

    status, response = getSession()
    auth_header = zsession.auth_token
    status, userResponse = zsession.get_request('/api/v1/users/name/emerson2@zededa.com')
    url_ext = "/api/v1/sessions/refresh"
    payload = {
        'userId': userResponse['id']
    }
    status, resp = zsession.put_request(url_ext, payload)
    print(f"{status}")

def createNewToken():

    status, response = getSession()
    status, userResponse = zsession.get_request('/api/v1/users/name/emerson2@zededa.com')
    url = "/api/v1/sessions/token/self"
    payload = {
        'expires': '7200'
    }
    status, resp = zsession.post_request(url, payload)
    status, response = getSession()

def getDevices():

    url_ext = "/api/v1/projects"
    status, response = zsession.get_request(url_ext)

def main():
    global zsession

    baseUrl = "https://zedcloud.canary.zededa.net"
    config = {
        'username': 'sathiyadev_canary@zededa.com',
        'password': 'Passw0rd@123',
        'auth_token': ''
    }
    zsession = zapi(baseUrl, config)
    login()
    #createNewToken()
    #refreshSession()
    getSession()


if __name__ == '__main__':
    main()
