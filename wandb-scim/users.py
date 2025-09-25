from __future__ import annotations

from typing import Any, TypedDict

import requests
from requests.auth import HTTPBasicAuth


class User:
    def __init__(self, base_url: str, username: str, api_key: str):
        """
        Initialize User object with username and API key.

        Args:
            base_url (str): Host url.
            username (str): The username for authentication (use empty string for service accounts).
            api_key (str): The API key for authentication.
        """
        self.base_url = base_url

        self.request_kwargs = dict(
            headers={"Content-Type": "application/json"},
            # Encode the username and API key into a base64-encoded string for Basic Authentication
            # For service accounts, use ":api_key" format (empty username)
            auth=HTTPBasicAuth(username, api_key),
            # Request hook to automatically raise for non-2XX status codes
            hooks={"response": lambda rsp, *_, **__: rsp.raise_for_status()},
        )

    def create(self, request_payload: dict[str, str]) -> dict[str, str]:
        """
        Creates a new user.

        Args:
            request_payload (dict): The payload containing user data.

        Returns:
            dict: The created user resource or error message.
        """
        print("Creating the User")
        schemas = ["urn:ietf:params:scim:schemas:core:2.0:User"]
        data = {
            "schemas": schemas,
            "emails": [
                {
                    "primary": True,
                    "value": request_payload["email"],
                }
            ],
            "userName": request_payload["name"],
        }
        # Send a POST request to create the user
        url = f"{self.base_url}/scim/Users"
        try:
            response = requests.post(url, json=data, **self.request_kwargs)
        except requests.HTTPError:
            return {
                "error": f"User creation failed. Status code: {response.status_code}",
                "details": response.text,
            }
        return response.json()

    def get(self, user_id: str) -> dict[str, str]:
        """
        Retrieves user details.

        Args:
            user_id (str): user_id of the user.

        Returns:
            dict: User resource with ETag in meta.version or error message.
        """
        print("Getting the User")
        # Send a GET request to retrieve the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        try:
            response = requests.get(url, **self.request_kwargs)
        except requests.HTTPError:
            return {
                "error": f"Get user failed. Status code: {response.status_code}",
                "details": response.text,
            }

        result = response.json()

        # Store ETag if present for conditional updates
        if etag := response.headers.get("ETag"):
            result["_etag"] = etag
        return result

    def get_all(self, filter: str | None = None) -> dict[str, str]:
        """
        Retrieves details of all users in the organization.

        Args:
            filter (str): Optional SCIM filter (e.g., 'userName eq "john.doe"' or 'emails.value eq "john@example.com"').

        Returns:
            dict: User list or error message.
        """
        print("Getting all the Users in org")
        # Send a GET request to retrieve all users
        url = f"{self.base_url}/scim/Users"
        params = {"filter": filter} if filter else {}
        try:
            response = requests.get(url, params=params, **self.request_kwargs)
        except requests.HTTPError:
            return {
                "error": f"Get users failed. Status code: {response.status_code}",
                "details": response.text,
            }

    def activate(self, user_id: str) -> str:
        """
        Activates a user.

        Args:
            user_id (str): user_id of the user.

        Returns:
            str: A message indicating whether the user activation was successful or failed.
        """
        print("Activating the User")
        # Send a PATCH request to activate the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "replace",
                    "value": {"active": True},
                }
            ],
        }
        try:
            response = requests.patch(url, json=payload, **self.request_kwargs)
        except requests.HTTPError:
            if response.status_code == 404:
                return "User not found"
            return f"Failed to activate user. Status code: {response.status_code}"
        return "User activated successfully!"

    def deactivate(self, user_id: str) -> str:
        """
        Deactivates a user.

        Args:
            user_id (str): user_id of the user.

        Returns:
            str: A message indicating whether the user deactivation was successful or failed.
        """
        print("Deactivating the User")
        # Send a PATCH request to deactivate the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        payload = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "replace",
                    "value": {"active": False},
                }
            ],
        }
        try:
            response = requests.patch(url, json=payload, **self.request_kwargs)
        except requests.HTTPError:
            if response.status_code == 404:
                return "User not found"
            return f"Failed to deactivate user. Status code: {response.status_code}"
        return "User deactivated successfully!"

    def delete(self, user_id: str) -> str:
        """
        Delete a user.

        Args:
            user_id (str): user_id of the user.

        Returns:
            str: A message indicating whether the user deletion was successful or failed.
        """
        print("Delete the User")
        # Send a DELETE request to delete the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        try:
            response = requests.delete(url, **self.request_kwargs)
        except requests.HTTPError:
            if response.status_code == 404:
                return "User not found"
            return f"Failed to delete user. Status code: {response.status_code}"
        return "User deleted successfully!"

    class _OrgRoleDict(TypedDict):
        roleName: str

    def assign_org_role(self, user_id: str, request_payload: _OrgRoleDict) -> str:
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
                    "value": request_payload["roleName"],
                    "value": request_payload["roleName"],
                }
            ],
        }
        # Send a PATCH request to assign the role to the user
        url = f"{self.base_url}/scim/Users/{user_id}"
        try:
            response = requests.patch(url, json=data, **self.request_kwargs)
        except requests.HTTPError:
            if response.status_code == 404:
                return "User not found"
            return f"Failed to update user. Status code: {response.status_code}"

        # Get the updated resource data from the response
        updated_data = response.json()
        print("Updated Data:", updated_data)
        return "User updated successfully"

    class _TeamRoleDict(TypedDict):
        teamName: str
        roleName: str

    def assign_team_role(
        self, user_id: str, request_payload: _TeamRoleDict
    ) -> dict[str, str]:
        """
        Assigns a role to a user of the team.

        Args:
            user_id (str): user_id of the user.
            request_payload (dict): The payload containing role information.
                - 'teamName': The name of the team.
                - 'roleName': The role to assign.

        Returns:
            dict: Updated user resource or error message.
        """
        print("Assign a role to the User of the team")
        data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "replace",
                    "path": "teamRoles",
                    "value": [request_payload],
                }
            ],
        }
        # Send a PATCH request to assign the role to the user of the team
        url = f"{self.base_url}/scim/Users/{user_id}"
        try:
            response = requests.patch(url, json=data, **self.request_kwargs)
        except requests.HTTPError:
            if response.status_code == 404:
                return {"error": "User not found"}
            return {
                "error": f"Failed to update user. Status code: {response.status_code}",
                "details": response.text,
            }
        return response.json()

    class _RegistryRoleDict(TypedDict):
        registryName: str
        roleName: str

    def assign_registry_role(
        self, user_id: str, request_payload: _RegistryRoleDict
    ) -> dict[str, Any]:
        """
        Assigns a new registry role to a user.

        Args:
            user_id (str): user_id of the user.
            request_payload (dict): The payload containing registry role information.
                - 'registryName': The name of the registry.
                - 'roleName': The role to assign.

        Returns:
            dict: Updated user resource or error message.
        """
        print("Assign registry role(s) to the User")
        data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "add",
                    "path": "registryRoles",
                    "value": [request_payload],
                }
            ],
        }
        # Send a PATCH request to assign the role to the user of the registry
        url = f"{self.base_url}/scim/Users/{user_id}"
        try:
            response = requests.patch(url, json=data, **self.request_kwargs)
        except requests.HTTPError:
            if response.status_code == 404:
                return {"error": "User not found"}
            return {
                "error": f"Failed to update user. Status code: {response.status_code}",
                "details": response.text,
            }

    def remove_registry_role(self, user_id: str, registry_name: str) -> dict[str, Any]:
        """
        Removes a user from a registry.

        Args:
            user_id (str): user_id of the user.
            registry_name (str): The name of the registry.

        Returns:
            dict: Updated user resource or error message.
        """
        print("Remove the User's registry role")
        data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [
                {
                    "op": "remove",
                    "path": f'registryRoles[registryName eq \\"{registry_name}\\"]',
                }
            ],
        }
        # Send a PATCH request to assign the role to the user of the registry
        url = f"{self.base_url}/scim/Users/{user_id}"
        try:
            response = requests.patch(url, json=data, **self.request_kwargs)
        except requests.HTTPError:
            if response.status_code == 404:
                return {"error": "User not found"}
            return {
                "error": f"Failed to update user. Status code: {response.status_code}",
                "details": response.text,
            }
