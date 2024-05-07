# calling_module.py

import requests
import sys
sys.path.append('../')
from teams import Teams  # Assuming the Teams class is defined in teams.py

def create_team(teams, display_name, member_ids):
    """
    Creates a new team.

    Args:
        teams (Teams): An instance of the Teams class.
        display_name (str): The display name of the new team.
        member_ids (list): The ID of the member to be added to the team.
    """
    try:
        # Create a new team
        create_team_response = teams.create(
            request_payload={
                "displayName": display_name,
                "members": member_ids
            }
        )
        print(create_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_team(teams, team_id):
    """
    Retrieves details of a specific team.

    Args:
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to retrieve.
    """
    try:
        # Get team details
        get_team_response = teams.get(team_id)
        print(get_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_teams(teams):
    """
    Retrieves details of all teams in the organization.

    Args:
        teams (Teams): An instance of the Teams class.
    """
    try:
        # Get all teams in the organization
        get_all_teams_response = teams.get_all()
        print(get_all_teams_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def add_members(teams, team_id, member_ids):
    """
    Updates a team by adding members.

    Args:
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to update.
        member_ids (list): The IDs of the members to added to the team.
    """
    try:
        # Update team by adding members
        add_members_response = teams.add_members(
            team_id,
            request_payload={"value": member_ids}
        )
        print(add_members_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def remove_members(teams, team_id, member_ids):
    """
    Updates a team by removing members.

    Args:
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to update.
        member_ids (list): The IDs of the members to removed from the team.
    """
    try:
        # Update team by removing members
        remove_members_response = teams.remove_members(
            team_id,
            request_payload={"value": member_ids}
        )
        print(remove_members_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

if __name__ == "__main__":
    username = "your_username"
    api_key = "your_api_key"
    base_url = "https://example.com/api"   # Base URL of the API

    # Instantiate the Teams class with your credentials
    teams = Teams(base_url, username, api_key)

    # Test Functions
    get_all_teams(teams)
    # create_team(teams, "test-team", ["member_id"])
    # get_team(teams, "team_id")
    # add_members(teams, "team_id", ["member_id"])
    # remove_members(teams, "team_id", ["member_id"])