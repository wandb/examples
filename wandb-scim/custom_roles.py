import requests
import base64

class CustomRole(object):
    def __init__(self, username, api_key):
        """
        Initializes the CustomRole object with username and API key.

        Args:
            username (str): The username for authentication.
            api_key (str): The API key for authentication.
        """
        # Encode the username and API key into a base64-encoded string for Basic Authentication
        auth_str = f"{username}:{api_key}"
        auth_bytes = auth_str.encode('ascii')
        self.auth_token = base64.b64encode(auth_bytes).decode('ascii')

        # Create the authorization header for API requests
        self.authorization_header = f"Basic {self.auth_token}"

    def _create_custom_role(self, url, request_payload):
        """
        Creates a new custom role.

        Args:
            url (str): The URL for the custom role creation endpoint.
            request_payload (dict): The payload containing custom role data.
                It should contain the following keys:
                    - 'permissionJson': The permissions JSON for the custom role.
                    - 'inheritedFrom': The inheritance information for the custom role.

        Returns:
            str: A message indicating whether the custom role creation was successful or failed.
        """
        print("Creating the custom role")
        data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
            "name": "Sample custom role",
            "description": "A sample custom role for example",
            "permissions": [request_payload['permissionJson']],
            "inheritedFrom": request_payload['inheritedFrom']
        }
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 201:
            return "Custom role has been created!"
        return f"Custom role creation failed. Status code: {response.status_code}"

    def _get_custom_role(self, url):
        """
        Retrieves custom role details.

        Args:
            url (str): The URL for the custom role retrieval endpoint.

        Returns:
            str: A message containing custom role details or indicating failure.
        """
        print("Getting the custom role")

        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return f"Custom role details: {response.text}"
        return f"Get custom role failed. Status code: {response.status_code}"

    def _get_all_custom_role(self, url):
        """
        Retrieves details of all custom roles in the organization.

        Args:
            url (str): The URL for the endpoint to get all custom roles.

        Returns:
            str: A message containing details of all custom roles or indicating failure.
        """
        print("Getting all the custom roles in org")

        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return f"All the custom roles details: {response.text}"
        return f"Get all custom roles failed. Status code: {response.status_code}"

    def _add_permission(self, url, request_payload):
        """
        Adds permission to a custom role.

        Args:
            url (str): The URL for the endpoint to add permission to the custom role.
            request_payload (dict): The payload containing permission information.
                It should contain the following key:
                    - 'permissionJson': The permissions JSON to be added to the custom role.

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
                    "value": [request_payload['permissionJson']]
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
            return "Custom role updated successfully"

        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to update custom role. Status code: {response.status_code}"

    def _remove_permission(self, url, request_payload):
        """
        Removes permission from a custom role.

        Args:
            url (str): The URL for the endpoint to remove permission from the custom role.
            request_payload (dict): The payload containing permission information.
                It should contain the following key:
                    - 'permissionJson': The permissions JSON to be removed from the custom role.

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
                    "value": [request_payload['permissionJson']]
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
            return "Custom role updated successfully"

        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to update custom role. Status code: {response.status_code}"

    def _update_custom_role(self, url, request_payload):
        """
        Updates name and description of a custom role.

        Args:
            url (str): The URL for the endpoint to update the custom role.
            request_payload (dict): The payload containing role information.
                It should contain the following keys:
                    - 'roleName': The name of the custom role.
                    - 'roleDescription': The description of the custom role.
                    - 'inheritedFrom': The inheritance information for the custom role.

        Returns:
            str: A message indicating whether the custom role update was successful or failed.
        """
        print("Update name and description of custom role")
        data = {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Role"],
            "name": request_payload['roleName'],
            "description": request_payload['roleDescription'],
            "inheritedFrom": request_payload['inheritedFrom']
        }
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }

        response = requests.put(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "Custom role updated successfully"

        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to update custom role. Status code: {response.status_code}"

    def _delete_custom_role(self, url):
        """
        Deletes a custom role.

        Args:
            url (str): The URL for the endpoint to delete the custom role.

        Returns:
            str: A message indicating whether the custom role deletion was successful or failed.
        """
        print("Deleting custom role")

        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }

        response = requests.delete(url, headers=headers)

        if response.status_code == 204:
            return "Custom role deleted successfully"
        elif response.status_code == 404:
            return "Custom role not found"
        else:
            return f"Failed to delete custom role. Status code: {response.status_code}"
