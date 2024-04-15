# calling_module.py

import requests
import sys
sys.path.append('../')
from users import User  # Assuming the User class is defined in user_module.py

def create_user(base_url, user):
    try:
        # Create a new user
        create_user_response = user._create_user(
            url=f"{base_url}/users",
            request_payload={
                "email": "newuser@example.com",
                "name": "newuser"
            }
        )
        print(create_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_user(base_url, user):
    try:
        # Get user details
        get_user_response = user._get_user(f"{base_url}/users/newuser")
        print(get_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_user(base_url, user):
    try:

        # Get all users in the organization
        get_all_users_response = user._get_all_user(f"{base_url}/users")
        print(get_all_users_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def deactivate_user(base_url, user):
    try:
        # Deactivate a user
        deactivate_user_response = user._deactivate_user(f"{base_url}/users/newuser")
        print(deactivate_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def assign_role_user(base_url, user):
    try:
        # Assign a role to the user
        assign_role_response = user._assign_role_user(
            url=f"{base_url}/users/newuser",
            request_payload={"roleName": "admin"}
        )
        print(assign_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")


def assign_team_user(base_url, user):
    try:
       # Assign team role to a user
        assign_team_role_response = user._assign_role_team(
            url=f"{base_url}/users/username",
            request_payload={"roleName": "viewer", "teamName": "teamA"}
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
    user._create_user(base_url , request_payload)

