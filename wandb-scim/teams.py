import requests
import base64

class Teams(object):
    def __init__(self, username, api_key):

        auth_str = f"{username}:{api_key}"
        auth_bytes = auth_str.encode('ascii')
        self.auth_token = base64.b64encode(auth_bytes).decode('ascii')

        self.authorization_header = f"Basic {self.auth_token}"

    def _create_team(self, url, request_payload):
        print("Creating the team")
        data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "displayName": request_payload['displayName'],
                "members": [
                    {
                        "value": request_payload['member']
                    }
                ]
            }
        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }
        response = requests.post(url, json=data , headers=headers)

        if response.status_code == 201:
            return "team has been created!"
        return f"team creation failed. Status code: {response.status_code}"
    
    def _get_team(self, url):
        print("Getting the team")

        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return f"team details: {response.text}"
        return f"Get team failed. Status code: {response.status_code}"

    def _get_all_teams(self, url):
        print("Getting all the teams in org")

        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return f"teams details: {response.text}"
        return f"Get teams failed. Status code: {response.status_code}"

    def _add_team(self, url, request_payload):

        print("Update team")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "add",
                        "path": "members",
                        "value": [
                    {
                                "value": request_payload['value'],
                            }
                    ]
                    }
                ]
            }
        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "team updated successfully"

        elif response.status_code == 404:
            return "team not found"
        else:
            return f"Failed to update team. Status code: {response.status_code}"

    def _remove_team(self, url, request_payload):

        print("Update team")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "remove",
                        "path": "members",
                        "value": [
                    {
                                "value": request_payload['value'],
                            }
                    ]
                    }
                ]
            }
        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "team updated successfully"

        elif response.status_code == 404:
            return "team not found"
        else:
            return f"Failed to update team. Status code: {response.status_code}"
        