# calling_module.py

import requests
import sys
sys.path.append('../')
from users import User  # Assuming the User class is defined in user_module.py

def create_user(user, email, name):
    """
    Creates a new user.

    Args:
        user (User): An instance of the User class.
        email (str): The email address of the new user.
        name (str): The name of the new user.
    """
    try:
        # Create a new user
        create_user_response = user.create(
            request_payload={"email": email, "name": name}
        )
        print(create_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_user(user, user_id):
    """
    Retrieves details of a specific user.

    Args:
        user (User): An instance of the User class.
        user_id (str): The ID of the user to retrieve.
    """
    try:
        # Get user details
        get_user_response = user.get(user_id)
        print(get_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_users(user):
    """
    Retrieves details of all users in the organization.

    Args:
        user (User): An instance of the User class.
    """
    try:
        # Get all users in the organization
        get_all_users_response = user.get_all()
        print(get_all_users_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def activate_user(user, user_id):
    """
    Activates a user.

    Args:
        user (User): An instance of the User class.
        user_id (str): The ID of the user to activate.
    """
    try:
        # Activate a user
        activate_user_response = user.activate(user_id)
        print(activate_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def deactivate_user(user, user_id):
    """
    Deactivates a user.

    Args:
        user (User): An instance of the User class.
        user_id (str): The ID of the user to deactivate.
    """
    try:
        # Deactivate a user
        deactivate_user_response = user.deactivate(user_id)
        print(deactivate_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def delete_user(user, user_id):
    """
    Deletes a user.

    Args:
        user (User): An instance of the User class.
        user_id (str): The ID of the user to delete.
    """
    try:
        # Delete a user
        delete_user_response = user.delete(user_id)
        print(delete_user_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def assign_org_role_to_user(user, user_id, role_name):
    """
    Assigns a org-level role to a user.

    Args:
        user (User): An instance of the User class.
        user_id (str): The ID of the user to assign the role to.
        role_name (str): The name of the role to assign.
    """
    try:
        # Assign a role to the user
        assign_org_role_response = user.assign_org_role(
            user_id,
            request_payload={"roleName": role_name}
        )
        print(assign_org_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")


def assign_team_role_to_user(user, user_id, team_name, role_name):
    """
    Assigns a team-level role to a user.

    Args:
        user (User): An instance of the User class.
        user_id (str): The ID of the user to assign the team role to.
        team_name (str): The name of the team to assign the role from.
        role_name (str): The name of the role to assign.
    """
    try:
        # Assign team role to a user
        assign_team_role_response = user.assign_team_role(
            user_id,
            request_payload={"roleName": role_name, "teamName": team_name}
        )
        print(assign_team_role_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

 
if __name__ == "__main__":
    username = "your_username"
    api_key = "your_api_key"
    base_url = "https://example.com/api"

    # Instantiate the User class with your credentials
    user = User(base_url, username, api_key)

    # Test Functions
    get_all_users(user)
    # create_user(user, "test@example.com", "Test User")
    # get_user(user, "user_id")
    # deactivate_user(user, "user_id")
    # activate_user(user, "user_id")
    # delete_user(user, "user_id")
    # assign_org_role_to_user(user, "user_id", "role_name")
    # assign_team_role_to_user(user, "user_id", "team_name", "role_name")