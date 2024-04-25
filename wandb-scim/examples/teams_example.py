# calling_module.py

import requests
import sys
sys.path.append('../')
from teams import Teams  # Assuming the Teams class is defined in teams.py

def create_team(base_url, teams, display_name, member_id):
    """
    Creates a new team.

    Args:
        base_url (str): The base URL of the API.
        teams (Teams): An instance of the Teams class.
        display_name (str): The display name of the new team.
        member_id (str): The ID of the member to be added to the team.
    """
    try:
        # Create a new team
        create_team_response = teams._create_team(
            url=f"{base_url}/Groups",
            request_payload={
                "displayName": display_name,
                "member": member_id
            }
        )
        print(create_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_team(base_url, teams, team_id):
    """
    Retrieves details of a specific team.

    Args:
        base_url (str): The base URL of the API.
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to retrieve.
    """
    try:
        # Get team details
        get_team_response = teams._get_team(f"{base_url}/Groups/{team_id}")
        print(get_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_teams(base_url, teams):
    """
    Retrieves details of all teams in the organization.

    Args:
        base_url (str): The base URL of the API.
        teams (Teams): An instance of the Teams class.
    """
    try:
        # Get all teams in the organization
        get_all_teams_response = teams._get_all_teams(f"{base_url}/Groups")
        print(get_all_teams_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def update_team_add_member(base_url, teams, team_id, member_id):
    """
    Updates a team by adding a member.

    Args:
        base_url (str): The base URL of the API.
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to update.
        member_id (str): The ID of the member to add to the team.
    """
    try:
        # Update team by adding a member
        update_team_response = teams._add_team(
            url=f"{base_url}/Groups/{team_id}",
            request_payload={"value": member_id}
        )
        print(update_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def update_team_remove_member(base_url, teams, team_id, member_id):
    """
    Updates a team by removing a member.

    Args:
        base_url (str): The base URL of the API.
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to update.
        member_id (str): The ID of the member to remove from the team.
    """
    try:
        # Update team by removing a member
        update_team_response = teams._remove_team(
            url=f"{base_url}/Groups/{team_id}",
            request_payload={"value": member_id}
        )
        print(update_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

if __name__ == "__main__":
    username = "your_username"
    api_key = "your_api_key"
    base_url = "https://localhost/api"  # Base URL of the API

    # Instantiate the Teams class with your credentials
    teams = Teams(username, api_key)

    # Test Functions
    get_all_teams(base_url, teams)
    # create_team(base_url, teams, "test-team", "member_id")
    # get_team(base_url, teams, "team_id")
    # update_team_add_member(base_url, teams, "team_id", "member_id")
    # update_team_remove_member(base_url, teams, "team_id", "member_id")