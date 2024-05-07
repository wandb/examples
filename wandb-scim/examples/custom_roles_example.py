# calling_module.py

import requests
import sys
sys.path.append('../')
from custom_roles import CustomRole  # Assuming CustomRole class is defined in custom_role.py

def create_custom_role(custom_role, permission_json, inherited_from):
    """
    Creates a new custom role.

    Args:
        custom_role (CustomRole): An instance of the CustomRole class.
        permission_json (str): JSON string representing permissions for the custom role.
        inherited_from (str): The source from which the custom role inherits permissions.
    """
    try:
        # Create a new custom role
        create_role_response = custom_role.create(
            request_payload={
                "permissionJson": permission_json,
                "inheritedFrom": inherited_from
            }
        )
        print(create_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_custom_role(custom_role, role_id):
    """
    Retrieves details of a specific custom role.

    Args:
        custom_role (CustomRole): An instance of the CustomRole class.
        role_id (str): The ID of the custom role to retrieve.
    """
    try:
        # Get details of a custom role
        get_role_response = custom_role.get(role_id)
        print(get_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_custom_roles(custom_role):
    """
    Retrieves details of all custom roles.

    Args:
        custom_role (CustomRole): An instance of the CustomRole class.
    """
    try:
        # Get all custom roles
        get_all_roles_response = custom_role.get_all()
        print(get_all_roles_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def add_permissions(custom_role, role_id, permission_json):
    """
    Adds permission to a custom role.

    Args:
        custom_role (CustomRole): An instance of the CustomRole class.
        role_id (str): The ID of the custom role to update.
        permission_json (str): JSON string representing the permission to add.
    """
    try:
        # Add permission to a custom role
        update_role_response = custom_role.add_permissions(
            role_id,
            request_payload={
                "permissionJson": permission_json
            }
        )
        print(update_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def remove_permissions(custom_role, role_id, permission_json):
    """
    Removes permission from a custom role.

    Args:
        custom_role (CustomRole): An instance of the CustomRole class.
        role_id (str): The ID of the custom role to update.
        permission_json (str): JSON string representing the permission to remove.
    """
    try:
        # Remove permission from a custom role
        remove_role_response = custom_role.remove_permissions(
            role_id,
            request_payload={
                "permissionJson": permission_json
            }
        )
        print(remove_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def update_custom_role(custom_role, role_id, update_params):
    """
    Updates a custom role.
    Args:
        custom_role (CustomRole): An instance of the CustomRole class.
        role_id (str): The ID of the custom role to update.
        update_params (dict): a dict which can contain any of the following values.
            name (str): The updated name of the custom role.
            description (str): The updated description of the custom role.
            inherited_from (str): The updated source from which the custom role inherits permissions.
    """
    try:
        request_payload={}
        if('name' in update_params):
            request_payload["name"] =  update_params['name']
        if('description' in update_params):
            request_payload["description"] =  update_params['description']
        if('inherited_from' in update_params):
            request_payload["inheritedFrom"] =  update_params['inherited_from']
        # Update a custom role
        update_custom_role_response = custom_role.update(
            role_id,
            request_payload
        )
        print(update_custom_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def delete_custom_role(custom_role, role_id):
    """
    Deletes a custom role.

    Args:
        custom_role (CustomRole): An instance of the CustomRole class.
        role_id (str): The ID of the custom role to delete.
    """
    try:
        # Delete the custom role
        delete_role_response = custom_role.delete(role_id)
        print(delete_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

if __name__ == "__main__":
    username = "your_username"
    api_key = "your_api_key"
    base_url = "https://example.com/api"  # Base URL of the API

    # Instantiate the CustomRole class with your credentials
    custom_role = CustomRole(base_url, username, api_key)
    role_id = "abc"  # Replace with actual role ID
    permission_json = {"name": "project:update"} # Replace with required permissions

    # Test Functions
    get_all_custom_roles(custom_role)
    # create_custom_role(custom_role, permission_json, "member")
    # get_custom_role(custom_role, role_id)
    # add_permissions(custom_role, role_id, permission_json)
    # remove_permissions(custom_role, role_id, permission_json)
    # update_custom_role(custom_role, role_id, {"name": "test-role", "description":"sample test role description", "inherited_from":"member"})
    # delete_custom_role(custom_role, role_id)