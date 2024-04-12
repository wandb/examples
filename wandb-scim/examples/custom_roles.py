import requests
from custom_roles import CustomRole  # Assuming CustomRole class is defined in custom_role.py

def create_custom_role(base_url, custom_role):
    try:
        # Create a new custom role
        create_role_response = custom_role._create_custom_role(
            url=f"{base_url}/Roles",
            request_payload={
                "permissionjson": { "name": "project:update" },
                "inheritedFrom": "viewer"
            }
        )
        print(create_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_custom_role(base_url, custom_role):
    try:
        # Get details of a custom role
        get_role_response = custom_role._get_custom_role(f"{base_url}/Roles/abc")  # Replace "abc" with the role ID
        print(get_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_roles(base_url, custom_role):
    try:
        # Get all custom roles
        get_all_roles_response = custom_role._get_all_custom_role(f"{base_url}/Roles")
        print(get_all_roles_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def add_permission(base_url, custom_role):
    try:
        # Add permission to a custom role
        update_role_response = custom_role._add_permission(
            url=f"{base_url}/Roles/abc",  # Replace "abc" with the role ID
            request_payload={
                "permissionJson":  { "name": "project:delete" }
            }
        )
        print(update_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def remove_permission(base_url, custom_role):
    try:
        # Remove permission from a custom role
        remove_role_response = custom_role._remove_permission(
            url=f"{base_url}/Roles/abc",  # Replace "abc" with the role ID
            request_payload={
                "permissionJson":  { "name": "project:delete" }
            }
        )
        print(remove_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def update_custom_role(base_url, custom_role):
    try:
        # Remove permission from a custom role
        update_custom_role_response = custom_role._update_custom_role(
            url=f"{base_url}/Roles/abc",  # Replace "abc" with the role ID
            request_payload={
                "roleName": "test-role",
                "roleDescription": "sample test role description",
                "inheritedFrom": "member" #member or viewer
            }
        )
        print(update_custom_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

if __name__ == "__main__":
    username = "your_username"
    api_key = "your_api_key"
    base_url = "https://example.com/api"  # Base URL of the API

    # Instantiate the CustomRole class with your credentials
    custom_role = CustomRole(username, api_key)




