import requests
from requests import Session
from requests.exceptions import ConnectionError
import json, uuid
from os import environ

class zapi(object):

    def __init__(self, base_url, config):

        self.base_url = base_url
        if "auth_token" in config:
            self.auth_token = config.get("auth_token")
        else:
            self.auth_token = None
        self.username = config['username']
        self.password = config['password']
        self.headers = {
            'content-type': 'application/json',
            'Authorization': "bearer {}".format(self.auth_token),
            'userAgent': 'sathiyadev-testing'
        }
        self.session = Session()
        self.x_csrf_token = self._get_CSRF_token()

    def _get_CSRF_token(self):

        request_id = uuid.uuid4()
        headers = {
            'content-type': 'application/json',
            'X-Request-Id': request_id.hex
        }
        login_url = self.base_url + '/api/v1/login'
        response = self.session.request("get", login_url, headers=headers)
        self.headers['X-CSRF-Token'] = response.headers['X-Csrf-Token']

    def login(self):
        """
        Method to login
        and assign auth-token to class variable
        """
        data = {
            "usernameAtRealm": self.username,
            "password": self.password
        }

        login_url = self.base_url + '/api/v1/login'
        print(f"method:POST \n"
              f"URL: {login_url} \n"
              f"Headers: {self.headers}\n"
              f"payload: {json.dumps(data)}")

        response = self.session.request("post", login_url, headers=self.headers, data=json.dumps(data))
        if response.status_code == 200:
            print(f"login to {login_url} is successful with response code {response.status_code}")
            resp = response.json()
            self.auth_token = resp['token']['base64']
            self.headers['Authorization'] = "bearer {}".format(self.auth_token)
            self.headers['X-CSRF-Token'] = response.headers['X-Csrf-Token']
            return 0, self.auth_token
        else:
            print(f"login to {login_url} failed with response code {response.status_code}")
            return 1, "authorization failed"

    def get_request(self, url_extention, params=None):


        url = self.base_url + url_extention

        print(f"method:GET \n"
              f"URL: {url} \n"
              f"Request Headers: {self.headers}")
        try:
            if params == None:
                response = self.session.get(url, headers=self.headers)
            else:
                response = self.session.get(url, headers=self.headers, params=params)

            print(f"Response Code: {response.status_code}")
            print(f"Response body: {response.json()}\n\n")
            if response.status_code != 200:
                return 1, response.text

            json_response = response.json()
            self.headers['X-CSRF-Token'] = response.headers['X-CSRF-Token']
            return 0, json_response
        except ConnectionError as e:
            print("Connection error retry GET request {}".format(str(e)))
            return 1, f"connection error {e}"
        except Exception as e:
            print("GET method to {} failed with exception {}".\
                    format(url, e))
            return 1, str(e)


    def put_request(self, url_extention, payload=None,retry=3):

        url = self.base_url + url_extention
        print("PUT request to API {}".format(url))

        print(f"method:PUT \n"
              f"URL: {url} \n"
              f"Request Headers: {self.headers}\n"
              f"payload: {json.dumps(payload)}")
        try:
            if payload == None:
                response = self.session.put(url, headers=self.headers)
            else:
                response = self.session.put(url, headers=self.headers, data=json.dumps(payload))

            print(f"Response Code: {response.status_code}")
            print(f"Response body: {response.json()}\n")
            if response.status_code not in [200,202]:
                print("PUT method to url {} failed with response code {} text output {}".\
                           format(url, response.status_code, response.text))
                return 1, response.text

            json_response = response.json()
            self.headers['X-CSRF-Token'] = response.headers['X-CSRF-Token']
            return 0, json_response
        except ConnectionError as e:
            print("Connection error start retry {}".format(str(e)))
            return 1, f"connection error: {e}"
        except Exception as e:
            print("PUT method to {} failed with exception {}".\
                       format(url, e))
            return 1, str(e)

    def post_request(self, url_extention, payload, files=None):

        url = self.base_url + url_extention
        print("POST request to API {}".format(url))

        print(f"method:POST \n"
              f"URL: {url} \n"
              f"REquest Headers: {self.headers}\n"
              f"payload: {json.dumps(payload)}")

        try:
            if files == None:
                response = self.session.post(url, headers=self.headers, data=json.dumps(payload))
            else:
                response = self.session.post(url, headers=self.headers, data=json.dumps(payload),\
                                         files={'file': open('files', 'r')})

            print(f"Response Code: {response.status_code}")
            print(f"Response body: {response.json()}\n")
            if response.status_code == 409:
                return 1, response.text
            elif response.status_code not in [200, 201, 202]:
                return 1, response.text

            self.headers['X-CSRF-Token'] = response.headers['X-CSRF-Token']
            return 0, response.text
        except ConnectionError as e:
            print("Connection error retry POST request")
            return 1, f"connection error: {e}"
        except Exception as e:
            print("Exception during POST to API {}".format(e))
            return 1, str(e)

    def delete_request (self, url_extention):

        url = self.base_url + url_extention
        print("DELETE request to API {}".format(url))

        print(f"method:DELETE \n"
              f"URL: {url} \n"
              f"Request Headers: {self.headers}")

        try:
            response = self.session.delete(url, headers=self.headers)
            print(f"Response code {response}")
            print(f"Response Boby {rsponse.json}")
            if response.status_code not in [200]:
                print("DELETE method to API {} failed".format(url))
                return 1, response.text

            self.headers['X-CSRF-Token'] = response.headers['X-CSRF-Token']
            return 0, response.text
        except ConnectionError as e:
            print("Connection error retry DELETE request")
            print("{}".format(e))
        except Exception as e:
            print("Exception during DELETE to API {}".format(e))
            return 1, str(e)

