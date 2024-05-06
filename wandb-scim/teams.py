import base64
import json
import requests

class Teams(object):
    def __init__(self, base_url, username, api_key):
        """
        Initializes the Teams object with username and API key.

        Args:
            base_url (str): Host url.
            username (str): The username for authentication.
            api_key (str): The API key for authentication.
        """
        # Encode the username and API key into a base64-encoded string for Basic Authentication
        auth_str = f"{username}:{api_key}"
        auth_bytes = auth_str.encode('ascii')
        self.auth_token = base64.b64encode(auth_bytes).decode('ascii')

        # Create the authorization header for API requests
        self.authorization_header = f"Basic {self.auth_token}"

    def create(self, request_payload):
        """
        Creates a new team.

        Args:
            request_payload (dict): The payload containing team data.
                It should contain the following keys:
                    - 'displayName': The display name of the team.
                    - 'member': The member to be added to the team.

        Returns:
            str: A message indicating whether the team creation was successful or failed.
        """
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
        # Send a POST request to create the team
        url = f"{self.base_url}/scim/Groups"
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 201:
            return "Team has been created!"
        return f"Team creation failed. Status code: {response.status_code}"

    def get(self, team_id):
        """
        Retrieves team details.

        Args:
            team_id (str): team_id of the team_id.

        Returns:
            str: A message containing team details or indicating failure.
        """
        print("Getting the team")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a GET request to retrieve the team
        url = f"{self.base_url}/scim/Groups/{team_id}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return json.dumps(response.json, indent=4)
        return f"Get team failed. Status code: {response.status_code}"

    def get_all(self):
        """
        Retrieves details of all teams in the organization.

        Returns:
            str: A message containing details of all teams or indicating failure.
        """
        print("Getting all the teams in org")
        headers = {
            "Authorization": self.authorization_header,
            "Content-Type": "application/json"
        }
        # Send a GET request to retrieve all teams
        url = f"{self.base_url}/scim/Groups"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return json.dumps(response.json, indent=4)
        return f"Get teams failed. Status code: {response.status_code}"

    def update(self, team_id, request_payload):
        """
        Adds a member to the team.

        Args:
            team_id (str): team_id of the team_id.
            request_payload (dict): The payload containing member information.
                It should contain the following key:
                    - 'value': The value of the member to be added to the team.

        Returns:
            str: A message indicating whether the member addition was successful or failed.
        """
        print("Adding member to the team")
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
        # Send a PATCH request to add the member to the team
        url = f"{self.base_url}/scim/Groups/{team_id}"
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "Team updated successfully"

        elif response.status_code == 404:
            return "Team not found"
        else:
            return f"Failed to update team. Status code: {response.status_code}"

    def remove(self, team_id, request_payload):
        """
        Removes a member from the team.

        Args:
            team_id (str): team_id of the team_id.
            request_payload (dict): The payload containing member information.
                It should contain the following key:
                    - 'value': The value of the member to be removed from the team.

        Returns:
            str: A message indicating whether the member removal was successful or failed.
        """
        print("Removing member from the team")
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
        # Send a PATCH request to remove the member from the team
        url = f"{self.base_url}/scim/Groups/{team_id}"
        response = requests.patch(url, json=data, headers=headers)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return "Team updated successfully"

        elif response.status_code == 404:
            return "Team not found"
        else:
            return f"Failed to update team. Status code: {response.status_code}"
