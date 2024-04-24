import requests
import sys
sys.path.append('../')
from custom_roles import CustomRole  # Assuming CustomRole class is defined in custom_role.py

def create_custom_role(base_url, custom_role, permission_json, inherited_from):
    try:
        # Create a new custom role
        create_role_response = custom_role._create_custom_role(
            url=f"{base_url}/Roles",
            request_payload={
                "permissionjson": permission_json,
                "inheritedFrom": inherited_from
            }
        )
        print(create_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_custom_role(base_url, custom_role, role_id):
    try:
        # Get details of a custom role
        get_role_response = custom_role._get_custom_role(f"{base_url}/Roles/{role_id}")
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

def add_permission(base_url, custom_role, role_id, permission_json):
    try:
        # Add permission to a custom role
        update_role_response = custom_role._add_permission(
            url=f"{base_url}/Roles/{role_id}",
            request_payload={
                "permissionJson": permission_json
            }
        )
        print(update_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def remove_permission(base_url, custom_role, role_id, permission_json):
    try:
        # Remove permission from a custom role
        remove_role_response = custom_role._remove_permission(
            url=f"{base_url}/Roles/{role_id}",
            request_payload={
                "permissionJson": permission_json
            }
        )
        print(remove_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def update_custom_role(base_url, custom_role, role_id, role_name, role_description, inherited_from):
    try:
        # Update a custom role
        update_custom_role_response = custom_role._update_custom_role(
            url=f"{base_url}/Roles/{role_id}",
            request_payload={
                "roleName": role_name,
                "roleDescription": role_description,
                "inheritedFrom": inherited_from
            }
        )
        print(update_custom_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def delete_custom_role(base_url, custom_role, role_id):
    try:
        # Delete the custom role
        delete_role_response = custom_role._delete_custom_role(
            url=f"{base_url}/Roles/{role_id}"
        )
        print(delete_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

if __name__ == "__main__":
    username = "your_username"
    api_key = "your_api_key"
    base_url = "https://example.com/api"  # Base URL of the API

    # Instantiate the CustomRole class with your credentials
    custom_role = CustomRole(username, api_key)
    get_all_roles(base_url, custom_role)
