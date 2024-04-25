# calling_module.py

import requests
import sys
sys.path.append('../')
from users import User  # Assuming the User class is defined in user_module.py

def create_user(base_url, user, email, name):
    """
    Creates a new user.

    Args:
        base_url (str): The base URL of the API.
        user (User): An instance of the User class.
        email (str): The email address of the new user.
        name (str): The name of the new user.
    """
    try:
        # Create a new user
        create_user_response = user._create_user(
            url=f"{base_url}/Users",
            request_payload={"email": email, "name": name}
        )
        print(create_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_user(base_url, user, user_id):
    """
    Retrieves details of a specific user.

    Args:
        base_url (str): The base URL of the API.
        user (User): An instance of the User class.
        user_id (str): The ID of the user to retrieve.
    """
    try:
        # Get user details
        get_user_response = user._get_user(f"{base_url}/Users/{user_id}")
        print(get_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_users(base_url, user):
    """
    Retrieves details of all users in the organization.

    Args:
        base_url (str): The base URL of the API.
        user (User): An instance of the User class.
    """
    try:
        # Get all users in the organization
        get_all_users_response = user._get_all_user(f"{base_url}/Users")
        print(get_all_users_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def deactivate_user(base_url, user, user_id):
    """
    Deactivates a user.

    Args:
        base_url (str): The base URL of the API.
        user (User): An instance of the User class.
        user_id (str): The ID of the user to deactivate.
    """
    try:
        # Deactivate a user
        deactivate_user_response = user._deactivate_user(f"{base_url}/Users/{user_id}")
        print(deactivate_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def assign_role_user(base_url, user, user_id, role_name):
    """
    Assigns a role to a user.

    Args:
        base_url (str): The base URL of the API.
        user (User): An instance of the User class.
        user_id (str): The ID of the user to assign the role to.
        role_name (str): The name of the role to assign.
    """
    try:
        # Assign a role to the user
        assign_role_response = user._assign_role_user(
            url=f"{base_url}/Users/{user_id}",
            request_payload={"roleName": role_name}
        )
        print(assign_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")


def assign_team_user(base_url, user, user_id, team_name, role_name):
    """
    Assigns a team role to a user.

    Args:
        base_url (str): The base URL of the API.
        user (User): An instance of the User class.
        user_id (str): The ID of the user to assign the team role to.
        team_name (str): The name of the team to assign the role from.
        role_name (str): The name of the role to assign.
    """
    try:
        # Assign team role to a user
        assign_team_role_response = user._assign_role_team(
            url=f"{base_url}/Users/{user_id}",
            request_payload={"roleName": role_name, "teamName": team_name}
        )
        print(assign_team_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

 
if __name__ == "__main__":
    username = "your_username"
    api_key = "your_api_key"
    base_url = "https://localhost/scim"

    # Instantiate the User class with your credentials
    user = User(username, api_key)
    # Retrieve details of all users in the organization
    get_all_users(base_url, user)
