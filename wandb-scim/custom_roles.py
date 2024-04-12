import requests
import base64

class CustomRole(object):
    def __init__(self, username, api_key):
        auth_str = f"{username}:{api_key}"
        auth_bytes = auth_str.encode('ascii')
        self.auth_token = base64.b64encode(auth_bytes).decode('ascii')

        self.authorization_header = f"Basic {self.auth_token}"

    def _create_custom_role(self, url, request_payload):
        print("Creating the custom role")
        data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
                "name": "Sample custom role",
                "description": "A sample custom role for example",
                "permissions": [ request_payload.permissionjson],
                "inheritedFrom": request_payload.inheritedFrom
                }
        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 201:
            return("custom role has been created!")
        return(f"custom role creation failed. Status code: {response.status_code}")
    
    def _get_custom_role(self, url):
        print("Getting the custom role")

        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return(f"custom role detials: {response.text}")
        return(f"Get custom role failed. Status code: {response.status_code}")

    def _get_all_custom_role(self, url):
        print("Getting all the custom roles in org")

        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.get(url,headers=headers)

        if response.status_code == 200:
            return(f"all the custom roles detials: {response.text}")
        return(f"Get all custom roles failed. Status code: {response.status_code}")

    def _add_permission(self, url, request_payload):

        print("add permission to a custom role")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "add", 
                        "path": "permissions",
                        "value": [ request_payload.permissionJson]
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
            return("custom role updated successfully")

        elif response.status_code == 404:
            return("custom role not found")
        else:
            return(f"Failed to update custom role. Status code: {response.status_code}")

    def _remove_permission(self, url, request_payload):

        print("remove permission to a custom role")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "remove", 
                        "path": "permissions",
                        "value": [ request_payload.permissionJson]
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
            return("custom role updated successfully")

        elif response.status_code == 404:
            return("custom role not found")
        else:
            return(f"Failed to update custom role. Status code: {response.status_code}")
        
    def _update_custom_role(self, url, request_payload):

        print("update name and description custom role")
        data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
                "name": request_payload.roleName,
                "description": request_payload.roleDescription,
                "inheritedFrom": request_payload.inheritedFrom 
                # inheritedFrom can either be member or viewer. 
            }
        headers = {
        "Authorization": self.authorization_header,
        "Content-Type": "application/json"
        }

        response = requests.put(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return("custom role updated successfully")

        elif response.status_code == 404:
            return("custom role not found")
        else:
            return(f"Failed to update custom role. Status code: {response.status_code}")