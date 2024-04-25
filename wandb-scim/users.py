import requests
import base64

class User(object):
    def __init__(self, username, api_key):
        """
        Initialize User object with username and API key.

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

    def _create_user(self, url, request_payload):
        """
        Creates a new user.

        Args:
            url (str): The URL for the user creation endpoint.
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
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 201:
            return "User has been created!"
        return f"User creation failed. Status code: {response.status_code}"
    
    def _get_user(self, url):
        """
        Retrieves user details.

        Args:
            url (str): The URL for the user retrieval endpoint.

        Returns:
            str: A message containing user details or indicating failure.
        """
        print("Getting the User")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a GET request to retrieve the user
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return f"user details: {response.text}"
        return f"Get user failed. Status code: {response.status_code}"

    def _get_all_user(self, url):
        """
        Retrieves details of all users in the organization.

        Args:
            url (str): The URL for the endpoint to get all users.

        Returns:
            str: A message containing details of all users or indicating failure.
        """
        print("Getting all the Users in org")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a GET request to retrieve all users
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return f"users details: {response.text}"
        return f"Get users failed. Status code: {response.status_code}"

    def _deactivate_user(self, url):
        """
        Deactivates a user.

        Args:
            url (str): The URL for the user deactivation endpoint.

        Returns:
            str: A message indicating whether the user deactivation was successful or failed.
        """
        print("deleting the User")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a DELETE request to deactivate the user
        response = requests.delete(url, headers=headers)

        if response.status_code == 204:
            return "User has deleted successfully!"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to delete user. Status code: {response.status_code}"

    def _assign_role_user(self, url, request_payload):
        """
        Assigns a role to a user.

        Args:
            url (str): The URL for the endpoint to assign a role to the user.
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
        response = requests.patch(url, json=data, headers=headers)
        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "User updated successfully"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to update user. Status code: {response.status_code}"

    def _assign_role_team(self, url, request_payload):
        """
        Assigns a role to a user of the team.

        Args:
            url (str): The URL for the endpoint to assign a role to the user of the team.
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
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "User updated successfully"
        elif response.status_code == 404:
            return "User not found"
        else:
            return f"Failed to update user. Status code: {response.status_code}"
