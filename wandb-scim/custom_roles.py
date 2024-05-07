import base64
import json
import requests

class CustomRole(object):
    def __init__(self, base_url, username, api_key):
        """
        Initializes the CustomRole object with username and API key.

        Args:
            base_url (str): Host url.
            username (str): The username for authentication.
            api_key (str): The API key for authentication.
        """
        # Encode the username and API key into a base64-encoded string for Basic Authentication
        auth_str = f"{username}:{api_key}"
        auth_bytes = auth_str.encode('ascii')
        self.base_url = base_url
        self.auth_token = base64.b64encode(auth_bytes).decode('ascii')

        # Create the authorization header for API requests
        self.authorization_header = f"Basic {self.auth_token}"

    def create(self, request_payload):
        """
        Creates a new custom role.

        Args:
            request_payload (dict): The payload containing custom role data.
                It should contain the following keys:
                    - 'permissions' (list[dict]): The permissions object for the custom role.
                    - 'inheritedFrom' (str): The inheritance information for the custom role.

        Returns:
            str: A message indicating whether the custom role creation was successful or failed.
        """
        print("Creating the custom role")
        data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
            "name": "Sample custom role",
            "description": "A sample custom role for example",
            "permissions": request_payload['permissions'],
            "inheritedFrom": request_payload['inheritedFrom']
        }
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/scim/Roles"
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 201:
            return "Custom role has been created!"
        return f"Custom role creation failed. Status code: {response.status_code}"

    def get(self, role_id):
        """
        Retrieves custom role details for role_id.

        Args:
            role_id (str): role_id from the custom role.

        Returns:
            str: A message containing custom role details or indicating failure.
        """
        print("Getting the custom role")

        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/scim/Roles/{role_id}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return json.dumps(response.text, indent=4)
        return f"Get custom role failed. Status code: {response.status_code}"

    def get_all(self):
        """
        Retrieves details of all custom roles in the organization.

        Returns:
            str: A message containing details of all custom roles or indicating failure.
        """
        print("Getting all the custom roles in org")

        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/scim/Roles"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return json.dumps(response.text, indent=4)
        return f"Get all custom roles failed. Status code: {response.status_code}"

    def add_permissions(self, role_id, request_payload):
        """
        Adds permission to a custom role.

        Args:
            role_id (str): role_id from the custom role.
            request_payload (dict): The payload containing permission information.
                It should contain the following key:
                    - 'permissions' (list[dict]): The permissions object to be added to the custom role.

        Returns:
            str: A message indicating whether the permission addition was successful or failed.
        """
        print("Add permission to a custom role")
        data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "add",
                    "path": "permissions",
                    "value": request_payload['permissions']
                }
            ]
        }
        url = f"{self.base_url}/scim/Roles/{role_id}"
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }

        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "Custom role updated successfully"

        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to update custom role. Status code: {response.status_code}"

    def remove_permissions(self, role_id, request_payload):
        """
        Removes permission from a custom role.

        Args:
            role_id (str): role_id from the custom role.
            request_payload (dict): The payload containing permission information.
                It should contain the following key:
                    - 'permissions' (list[dict]): The permissions Object to be removed from the custom role.

        Returns:
            str: A message indicating whether the permission removal was successful or failed.
        """
        print("Remove permission from a custom role")
        data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "remove",
                    "path": "permissions",
                    "value": request_payload['permissions']
                }
            ]
        }
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/scim/Roles/{role_id}"
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "Custom role updated successfully"

        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to update custom role. Status code: {response.status_code}"

    def update(self, role_id, request_payload):
        """
        Updates name and description of a custom role.

        Args:
            role_id (str): role_id from the custom role.
            request_payload (dict): The payload containing role information.
                It should contain any of the following keys:
                    - 'roleName' (str): The name of the custom role.
                    - 'roleDescription' (str): The description of the custom role.
                    - 'inheritedFrom' (str): The inheritance information for the custom role.

        Returns:
            str: A message indicating whether the custom role update was successful or failed.
        """
        print("Update name and description of custom role")
        data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"]
        }
        data.update(request_payload)
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/scim/Roles/{role_id}"
        response = requests.put(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "Custom role updated successfully"

        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to update custom role. Status code: {response.status_code}"

    def delete(self, role_id):
        """
        Deletes a custom role.

        Args:
            role_id (str): role_id from the custom role.

        Returns:
            str: A message indicating whether the custom role deletion was successful or failed.
        """
        print("Deleting custom role")

        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/scim/Roles/{role_id}"
        response = requests.delete(url, headers=headers)

        if response.status_code == 204:
            return "Custom role deleted successfully"
        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to delete custom role. Status code: {response.status_code}"
