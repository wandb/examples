# calling_module.py

import requests
import sys
sys.path.append('../')
from users import User  # Assuming the User class is defined in user_module.py

def create_user(base_url, user, email, name):
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
    try:
        # Get user details
        get_user_response = user._get_user(f"{base_url}/Users/{user_id}")
        print(get_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_users(base_url, user):
    try:
        # Get all users in the organization
        get_all_users_response = user._get_all_user(f"{base_url}/Users")
        print(get_all_users_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def deactivate_user(base_url, user, user_id):
    try:
        # Deactivate a user
        deactivate_user_response = user._deactivate_user(f"{base_url}/Users/{user_id}")
        print(deactivate_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def assign_role_user(base_url, user, user_id, role_name):
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
    get_all_users(base_url, user)
