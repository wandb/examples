import requests

class Teams(object):
    # def __init__(self, common_params):
    #     dbcli_apiclient = ApiClient(common_params["api_user"], password=common_params["api_password"],
    #                             host='https://accounts.cloud.databricks.com', 
    #                             verify=True, command_name='Python Dev')
    #     self.accounts_api_client = AccountsApi(dbcli_apiclient)

    def _create_team(self, url, request_payload):
        print("Creating the team")
        data = {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
                "displayName": request_payload.displayName,
                "members": [
                    {
                        "value": request_payload.value
                    }
                ]
            }
        response = requests.post(url, json=data)
        
        if response.status_code == 201:
            return("team has been created!")
        return(f"team creation failed. Status code: {response.status_code}")
    
    def _get_team(self, url):
        print("Getting the team")

        response = requests.get(url)

        if response.status_code == 200:
            return(f"team detials: {response.text}")
        return(f"Get team failed. Status code: {response.status_code}")

    def _get_all_teams(self, url):
        print("Getting all the teams in org")

        response = requests.get(url)

        if response.status_code == 200:
            return(f"teams detials: {response.text}")
        return(f"Get teams failed. Status code: {response.status_code}")

    def _add_team(self, url, request_payload):

        print("update team")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "add",
                        "path": "members",
                        "value": [
                    {
                                "value": request_payload.value,
                            }
                    ]
                    }
                ]
            }

        response = requests.patch(url, json=data)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return("team updated successfully")

        elif response.status_code == 404:
            return("team not found")
        else:
            return(f"Failed to update team. Status code: {response.status_code}")

    def _remove_team(self, url, request_payload):

        print("update team")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "remove",
                        "path": "members",
                        "value": [
                    {
                                "value": request_payload.value,
                            }
                    ]
                    }
                ]
            }

        response = requests.patch(url, json=data)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return("team updated successfully")

        elif response.status_code == 404:
            return("team not found")
        else:
            return(f"Failed to update team. Status code: {response.status_code}")
        