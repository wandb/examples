import requests

class User(object):
    # def __init__(self, common_params):
    #     dbcli_apiclient = ApiClient(common_params["api_user"], password=common_params["api_password"],
    #                             host='https://accounts.cloud.databricks.com', 
    #                             verify=True, command_name='Python Dev')
    #     self.accounts_api_client = AccountsApi(dbcli_apiclient)

    def _create_user(self, url, request_payload):
        print("Creating the User")
        data = {
                "schemas": [
                    "urn:ietf:params:scim:schemas:core:2.0:User"
                ],
                "emails": [
                    {
                    "primary": true,
                    "value": request_payload.email
                    }
                ],
                "userName": request_payload.name
                }
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return("User has been created!")
        return(f"User creation failed. Status code: {response.status_code}")
    
    def _get_user(self, url):
        print("Getting the User")

        response = requests.get(url)

        if response.status_code == 200:
            return(f"user detials: {response.text}")
        return(f"Get user failed. Status code: {response.status_code}")

    def _get_all_user(self, url):
        print("Getting all the Users in org")

        response = requests.get(url)

        if response.status_code == 200:
            return(f"users detials: {response.text}")
        return(f"Get users failed. Status code: {response.status_code}")

    def _deactivate_user(self, url):
        print("deleting the User")

        response = requests.delete(url)

        if response.status_code == 204:
            return("user deleted successfully")
        elif response.status_code == 404:
            return("user not found")
        else:
            return(f"Failed to delete user. Status code: {response.status_code}")

    def _assign_role(self, url, request_payload):
        # request_payload.role It can be one of admin, viewer or member.
        print("assign a role to the User")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "replace",
                        "path": "organizationRole",
                        "value": request_payload.roleName
                    }
                ]
            }

        response = requests.patch(url, json=data)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return("user updated successfully")

        elif response.status_code == 404:
            return("user not found")
        else:
            return(f"Failed to update user. Status code: {response.status_code}")

    def _assign_role(self,url, request_payload):
        print("assign a role to the User of the team")
        data = {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {
                        "op": "replace",
                        "path": "teamRoles",
                        "value": [
                            {
                                "roleName": request_payload.roleName,
                                "teamName": request_payload.teamName
                            }
                        ]
                    }
                ]
            }

        response = requests.patch(url, json=data)

        if response.status_code == 200:
            updated_data = response.json()  # Get the updated resource data from the response
            print("Updated Data:", updated_data)
            return("user updated successfully")
        elif response.status_code == 404:
            return("user not found")
        else:
            return(f"Failed to update user. Status code: {response.status_code}")