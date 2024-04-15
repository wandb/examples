import requests
import base64

class User(object):
    def __init__(self, username, api_key):
        auth_str = f"{username}:{api_key}"
        auth_bytes = auth_str.encode('ascii')
        self.auth_token = base64.b64encode(auth_bytes).decode('ascii')

        self.authorization_header = f"Basic {self.auth_token}"

    def _create_user(self, url, request_payload):
        print("Creating the User")
        data = {
                "schemas": [
                    "urn:ietf:params:scim:schemas:core:2.0:User"
                ],
                "emails": [
                    {
                    "primary": True,
                    "value": request_payload['email']
                    }
                ],
                "userName": request_payload['name']
                }   
        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }
        # response = requests.post(url, json=data, headers=headers)
        response = {"status_code": 200} 
        if response['status_code'] == 200:
            return("User has been created!")
        return(f"User creation failed. Status code: {response['status_code']}")
    
    def _get_user(self, url):
        print("Getting the User")

        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }
        

        response = requests.get(url, headers=headers)

        if response['status_code'] == 200:
            return(f"user detials: {response['text']}")
        return(f"Get user failed. Status code: {response['status_code']}")

    def _get_all_user(self, url):
        print("Getting all the Users in org")

        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)

        if response['status_code'] == 200:
            return(f"users detials: {response['text']}")
        return(f"Get users failed. Status code: {response['status_code']}")

    def _deactivate_user(self, url):
        print("deleting the User")

        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.delete(url,headers=headers)

        if response['status_code'] == 204:
            return("user deleted successfully")
        elif response['status_code'] == 404:
            return("user not found")
        else:
            return(f"Failed to delete user. Status code: {response['status_code']}")

    def _assign_role_user(self, url, request_payload):
        # request_payload[role] It can be one of admin, viewer or member.
        print("assign a role to the User")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "replace",
                        "path": "organizationRole",
                        "value": request_payload['roleName']
                    }
                ]
            }
        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.patch(url, json=data, headers=headers)

        if response['status_code'] == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return("user updated successfully")

        elif response['status_code'] == 404:
            return("user not found")
        else:
            return(f"Failed to update user. Status code: {response['status_code']}")

    def _assign_role_team(self,url, request_payload):
        print("assign a role to the User of the team")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "replace",
                        "path": "teamRoles",
                        "value": [
                            {
                                "roleName": request_payload['roleName'],
                                "teamName": request_payload['roleName']
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

        if response['status_code'] == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return("user updated successfully")
        elif response['status_code'] == 404:
            return("user not found")
        else:
            return(f"Failed to update user. Status code: {response['status_code']}")