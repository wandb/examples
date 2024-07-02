import base64
import json
import requests

class User(object):
    def __init__(self, base_url, username, api_key):
        """
        Initialize User object with username and API key.

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
        Creates a new user.

        Args:
            request_payload (dict): The payload containing user data.

        Returns:
            str: A message indicating whether the user creation was successful or failed.
        """
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
        # Send a POST request to create the user
        url = f"{self.base_url}/scim/Users"
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 201:
            return "User has been created!"
        return f"User creation failed. Status code: {response.status_code}"
    
    def get(self, user_id):
        """
        Retrieves user details.

        Args:
            user_id (str): user_id of the user.

        Returns:
            str: A message containing user details or indicating failure.
        """
        print("Getting the User")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a GET request to retrieve the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return json.dumps(response.text, indent=4)
        return f"Get user failed. Status code: {response.status_code}"

    def get_all(self):
        """
        Retrieves details of all users in the organization.

        Returns:
            str: A message containing details of all users or indicating failure.
        """
        print("Getting all the Users in org")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        url = f"{self.base_url}/scim/Users"
        # Send a GET request to retrieve all users
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return json.dumps(response.text, indent=4)
        return f"Get users failed. Status code: {response.status_code}"

    def activate(self, user_id):
        """
        Activates a user.

        Args:
            user_id (str): user_id of the user.

        Returns:
            str: A message indicating whether the user activation was successful or failed.
        """
        print("Activating the User")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a PATCH request to activate the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "replace",
                    "value": {"active": True}
                }
            ]
        }
        response = requests.patch(url, headers=headers, json=payload)

        if response.status_code == 200:
            return "User activated successfully!"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to activate user. Status code: {response.status_code}"
    
    def deactivate(self, user_id):
        """
        Deactivates a user.

        Args:
            user_id (str): user_id of the user.

        Returns:
            str: A message indicating whether the user deactivation was successful or failed.
        """
        print("Deactivating the User")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a PATCH request to deactivate the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "replace",
                    "value": {"active": False}
                }
            ]
        }
        response = requests.patch(url, headers=headers, json=payload)

        if response.status_code == 200:
            return "User deactivated successfully!"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to deactivate user. Status code: {response.status_code}"

    def delete(self, user_id):
        """
        Delete a user.

        Args:
            user_id (str): user_id of the user.

        Returns:
            str: A message indicating whether the user deletion was successful or failed.
        """
        print("Delete the User")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a DELETE request to delete the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        response = requests.delete(url, headers=headers)

        if response.status_code == 204:
            return "User deleted successfully!"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to delete user. Status code: {response.status_code}"

    def assign_org_role(self, user_id, request_payload):
        """
        Assigns a role to a user.

        Args:
            user_id (str): user_id of the user.
            request_payload (dict): The payload containing role information.
                It should contain the following key:
                    - 'roleName': The role to be assigned to the user. It can be one of 'admin', 'viewer', or 'member'.

        Returns:
            str: A message indicating whether the role assignment was successful or failed.
        """
        print("Assign a role to the User")
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
        # Send a PATCH request to assign the role to the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "User updated successfully"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to update user. Status code: {response.status_code}"

    def assign_team_role(self, user_id, request_payload):
        """
        Assigns a role to a user of the team.

        Args:
            user_id (str): user_id of the user.
            request_payload (dict): The payload containing role information.

        Returns:
            str: A message indicating whether the role assignment was successful or failed.
        """
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
        # Send a PATCH request to assign the role to the user of the team
        url = f"{self.base_url}/scim/Users/{user_id}"
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "User updated successfully"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to update user. Status code: {response.status_code}"
