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
        member_object = []
        for id in member_ids:
            member_object.append({"value": id})
        create_team_response = teams.create(
            request_payload={
                "displayName": display_name,
                "members": member_object
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
        member_object = []
        for id in member_ids:
            member_object.append({"value": id})
        add_members_response = teams.add_members(
            team_id,
            member_object
        )
        print(add_members_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def remove_member(teams, team_id, member_id):
    """
    Updates a team by removing a specific member.

    Args:
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to update.
        member_id (str): The ID of the member to be removed from the team.
    """
    try:
        # Update team by removing a specific member
        remove_member_response = teams.remove_member(team_id, member_id)
        print(remove_member_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def remove_multiple_members(teams, team_id, member_ids):
    """
    Updates a team by removing multiple members (one API call per member).

    Args:
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to update.
        member_ids (list): The IDs of the members to be removed from the team.
    """
    try:
        # Remove each member individually since API only supports one operation at a time
        for member_id in member_ids:
            print(f"Removing member: {member_id}")
            remove_member_response = teams.remove_member(team_id, member_id)
            print(remove_member_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def remove_all_members(teams, team_id):
    """
    Removes all members from a team.

    Args:
        teams (Teams): An instance of the Teams class.
        team_id (str): The ID of the team to remove all members from.
    """
    try:
        # Remove all members from the team
        remove_all_response = teams.remove_all_members(team_id)
        print(remove_all_response)
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
    # create_team(teams, "team-name-asdfghjsdfgh",["VXNlcjoxNjg1NzA5"])
    # get_team(teams, "team_id")
    # add_members(teams, "RW50aXR5OjIyMTgyMjI=", ["VXNlcjoxODcxODU1", "VXNlcjoxNjg1NzA5"])
    # remove_member(teams, "RW50aXR5OjIyMTgyMjI=", "VXNlcjoxODcxODU1")  # Remove single member
    # remove_multiple_members(teams, "RW50aXR5OjIyMTgyMjI=", ["VXNlcjoxODcxODU1", "VXNlcjoxNjg1NzA5"])  # Remove multiple members
    # remove_all_members(teams, "RW50aXR1OjIyMTgyMjI=")  # Remove all members