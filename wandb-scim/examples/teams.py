import requests
from teams import Teams  # Assuming the Teams class is defined in teams.py

def create_team(base_url, teams):
    try:
        # Create a new team
        create_team_response = teams._create_team(
            url=f"{base_url}/Groups",
            request_payload={
                "displayName": "NewTeam",
                "value": "def"
            }
        )
        print(create_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_team(base_url, teams):
    try:
        # Get team details
        get_team_response = teams._get_team(f"{base_url}/Groups/123")  # Replace "123" with the team ID
        print(get_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def get_all_teams(base_url, teams):
    try:
        # Get all teams in the organization
        get_all_teams_response = teams._get_all_teams(f"{base_url}/Groups")
        print(get_all_teams_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def update_team_add_member(base_url, teams):
    try:
        # Update team by adding a member
        update_team_response = teams._add_team(
            url=f"{base_url}/Groups/123",  # Replace "123" with the team ID
            request_payload={
               "value" : "def"
            }
        )
        print(update_team_response)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during API request: {str(e)}")

def update_team_remove_member(base_url, teams):
    try:
        # Update team by removing a member
        update_team_response = teams._remove_team(
            url=f"{base_url}/Groups/123",  # Replace "123" with the team ID
            request_payload={
                 "value" : "def"
            }
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

    # Test different methods
    create_team(base_url, teams)

