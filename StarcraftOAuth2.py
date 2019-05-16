#USEFUL LINK TO MIMIC!: https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api


from oauthlib.oauth2 import WebApplicationClient
from oauthlib.oauth2 import Client

CLIENT_ID     = "8dc3b65ca4fa421eb4d18d5d7759b40e"
CLIENT_SECRET = "z1Z4Gg0Go7YvoFYsetCvrnH0t5mf3H1q"

GRANT_TYPE = "client_credentials"

AUTHORIZE_URI = "https://us.battle.net/oauth/authorize"
TOKEN_URI     = "https://us.battle.net/oauth/token"

#-----------------------------------------------------

# client = Client(CLIENT_ID)
# print("PREPARE AUTHORIZATION REQUEST")
# print(client.prepare_authorization_request(AUTHORIZE_URI))
# print("PREPARE TOKEN REQUEST")
# print(client.prepare_token_request(TOKEN_URI))

#-----------------------------------------------------

# client = WebApplicationClient(CLIENT_ID)
# #uri = 'https://example.com/callback?code=sdfkjh345&state=sfetw45'
# print("PREPARE REQUEST URI")
# print(client.prepare_request_uri("https://us.battle.net/oauth/token", client_secret=CLIENT_SECRET, grant_type=GRANT_TYPE))
# print("PARSE REQUEST URI RESPONSE")
# print(client.parse_request_uri_response("https://us.battle.net/oauth/token", state=None))
# #print(client.parse_request_uri_response(uri, state='other'))

###########################################################


import requests, json
import subprocess
import sys

#callback url specified when the application was defined
callback_uri = "<<your redirect_uri goes here>>"

test_api_url = "<<the URL of the API you want to call, along with any parameters, goes here>>"

#step A - simulate a request from a browser on the authorize_url - will return an authorization code after the user is
# prompted for credentials.

# authorization_redirect_url = AUTHORIZE_URI + '?response_type=code&client_id=' + CLIENT_ID + '&redirect_uri=' + callback_uri + '&scope=openid'
authorization_redirect_url = AUTHORIZE_URI + '?response_type=code&client_id=' + CLIENT_ID + '&redirect_uri=' + '&scope=openid'


print("go to the following url on the browser and enter the code from the returned url: ")
print("---  " + authorization_redirect_url + "  ---")
authorization_code = input('code: ')

# step I, J - turn the authorization code into a access token, etc
# data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': callback_uri}
data = {'grant_type': 'authorization_code', 'code': authorization_code, 'redirect_uri': ''}
print("requesting access token")
access_token_response = requests.post(TOKEN_URI, data=data, verify=False, allow_redirects=False, auth=(CLIENT_ID, CLIENT_SECRET))

print("response")
print(access_token_response.headers)
print('body: ' + access_token_response.text)

# we can now use the access_token as much as we want to access protected resources.
tokens = json.loads(access_token_response.text)
access_token = tokens['access_token']
print("access token: " + access_token)

api_call_headers = {'Authorization': 'Bearer ' + access_token}
api_call_response = requests.get(test_api_url, headers=api_call_headers, verify=False)

print(api_call_response.text)