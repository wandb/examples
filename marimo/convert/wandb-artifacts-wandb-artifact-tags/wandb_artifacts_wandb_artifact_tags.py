# /// script
# dependencies = ["wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/WandB_Artifact_Tags.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{artifact-tags} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{artifact-tags} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## W&B Artifact version tags

    W&B Models supports finding and retrieving artifacts using version tags from Registry or ML projects. Filtering directly by artifact version tags via the SDK offers a more efficient way to retrieve only the artifacts you need instead of grabbing every artifact from a collection and parsing them manually or using aliases which can present challenges due to enforced uniqueness within each collection.

    In this notebook, you will create multiple model versions in a collection in the Model registry. Each of the versions has been assigned multiple tags making discoverability and retrieval simple using the SDK.

    You can read more about organizing artifacts with tags in the <a href="https://docs.wandb.ai/guides/registry/organize-with-tags/" target="_parent">W&B documentation</a>.

    You can find Registry in your W&B account by clicking on the Registry link in the left sidebar in the Applications section.
    <br><br>
    <img src="https://rratshin-images.s3.us-west-2.amazonaws.com/colab_artifact_registry.png" width="800" alt="Weights & Biases" border="1" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisites

    Install the W&B Python SDK and log in:
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell
def _():
    # Log in to your W&B account
    import wandb
    wandb.login()
    return (wandb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize a W&B run

    Import additional Python libraries and initialize a W&B run to generate demo artifacts[link text](https://):
    """)
    return


@app.cell
def _(wandb):
    import random
    import math
    import pandas as pd
    import numpy as np
    import os

    PROJECT = "artifacts-example"
    JOB_TYPE = "generate_artifacts"

    run = wandb.init(
      project=PROJECT,
      job_type=JOB_TYPE
    )
    return np, pd, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate demo artifact versions with tags

    The following code block will generate four artifact versions for each AWS region. Each artifact version represents the result of fine-tuning a base model. The artifact versions will be tagged with the following:

    1. Region tag: Specifies the AWS region where the model is stored or deployed
    2. Base model tag: Indicates which base model was fine-tuned
    3. Status tag: Specifies the model's status, when applicable (e.g., production, candidate, archived)

    To create the artifact versions, simply execute the code block.
    """)
    return


@app.cell
def _(np, pd, run, wandb):
    regions = ['us-west-2', 'eu-central-1', 'me-central-1', 'ap-east-1']
    artifact_base_model_by_region = {'us-west-2': ['Llama 3 Instruct - 70B', 'Llama 3 Instruct - 70B', 'Gemini 1_5 Pro', 'Claude 3_5 Sonnet'], 'eu-central-1': ['Llama 3 Instruct - 70B', 'Gemini 1_5 Pro', 'Claude 3_5 Sonnet', 'Llama 3 Instruct - 70B'], 'me-central-1': ['Llama 3 Instruct - 70B', 'Gemini 1_5 Pro', 'Gemini 1_5 Pro', 'Claude 3_5 Sonnet'], 'ap-east-1': ['Claude 3_5 Sonnet', 'Gemini 1_5 Pro', 'Claude 3_5 Sonnet', 'Gemini 1_5 Pro']}
    artifact_metadata_by_region = {'us-west-2': [0.7, 0.73, 0.77, 0.71], 'eu-central-1': [0.76, 0.81, 0.79, 0.77], 'me-central-1': [0.77, 0.68, 0.74, 0.83], 'ap-east-1': [0.82, 0.79, 0.76, 0.76]}
    artifact_status_by_region = {'us-west-2': ['production', 'candidate', 'archived', 'archived'], 'eu-central-1': ['production', 'candidate', 'archived', 'archived'], 'me-central-1': ['production', 'candidate', 'candidate', 'archived'], 'ap-east-1': ['production', 'candidate', 'archived', 'archived']}
    for region in regions:
    # set up tag values
    # AWS regions (us-west-2 = Oregon, eu-central-1 = Frankfurt, me-central-1 = UAE, ap-east-1 = Hong Kong)
        i = 0
    # add base model tags to each artifact version
        for base_model in artifact_base_model_by_region[region]:
            df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
            df.to_json('test_model.pt', orient='records', lines=True)
            model_name = 'test_model_' + region + '_' + str(i)
            at = wandb.Artifact(name=model_name, type='model')
            at.add_file('test_model.pt')
    # add performance metadata to each artifact version
            arti = run.log_artifact(at)
            arti.wait()
            if i == 3:
                arti.tags = [region, base_model]
                arti.metadata = {'accuracy': artifact_metadata_by_region[region][i]}
            else:
    # add model status tags to each artifact version
                arti.tags = [region, base_model, artifact_status_by_region[region][i]]
                arti.metadata = {'accuracy': artifact_metadata_by_region[region][i]}
            arti.save()
            registered_at = run.link_artifact(at, f'wandb-registry-model/Artifact Demo Models')
            i = i + 1
    # create 4 artifact versions for each region
    # mark the run as finished
    run.finish()  # create random dataset to use as a model version  # assign a status tag to each artifact except the 4th which has no status tag assigned  # Provide one or more tags in a list  # link the artifact to the model registry
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filter artifacts using version tags

    After generating the 16 artifacts using the code block above, the Artifact Demo Models collection in your model registry should look like this:
    <br><br>

    <img src="https://rratshin-images.s3.us-west-2.amazonaws.com/colab_artifact_version_tags5.png" width="800" alt="Weights & Biases" border="1" />

    This colab presumes that you have not one production model in a collection, but multiple production models, one for each AWS region. In cases where only a single production model exists in a collection, using a unique alias to identify this model is generally the right approach. The enforced uniqueness of aliases makes them extremely valuable when searching for and retrieving a specific artifact version. Aliases can also be used in W&B Models to trigger automated workflows, or Automations, which are often used for model testing and deployment as part of a CI/CD pipeline. But event-based triggers are not always necessary and there are times when you want to track down and retrieve multiple artifact versions based on a search filter, such as when multiple production models exist. This is when tags are the answer.

    An example where version artifact tags come in handy is when deploying models using Amazon Sagemaker. The Amazon S3 bucket where the model artifacts are stored must be in the same AWS Region as the model that you are creating. In cases where it is a requirement to deploy specific models to specific regions, retrieving those models using an AWS region artifact tag ensures that the right model exists and that the right model is deployed.

    Attaching tags to artifact versions also helps with compliance requirements. Filtering by artifact tag retruns a specific model of interest and, from there, it is easy to track the detailed lineage of this model, including all input and output artifacts, using W&B Registry. For example, during an audit, it might be required to produce the exact dataset used for training the model deployed in the Central Europe region to ensure that it did not contain any PII data or other sensitive information.

    We have compiled a number of use cases that require filtering by artifact version tags to retrieve the right artifacts. To see version tag filtering using the SDK in action, just execute the following code block:
    """)
    return


@app.cell
def _(wandb):
    api = wandb.Api()
    # if you belong to multiple orgs, prefix the 'name' value in the api.artifacts call with the org you are fetching from:
    # name=f"{INSERT_ORG_NAME}/wandb-registry-model/Artifact Demo Models"

    ##########################################################
    ##### Artifact Version Tag Use Cases #####################
    ##########################################################

    # 1. As an ML Platform Engineer, I need to ensure the correct models are deployed in each region.
    #   Action: Filter by `production`, `us-west-2`, and base_model tags. (Base model should be `Llama 3 Instruct - 70B`).
    #   Scenario: Verify that only production-ready models are deployed in the US-West region, using the correct base model.
    #   Benefit: Ensures accurate deployment for specific regions and base models.

    print("\nArtifact Version Tag Use Case #1")
    print("As an ML Platform Engineer, I need to ensure the correct models are deployed in each region.")
    print("---------------------------------------\n")
    artifact_versions = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['production', 'us-west-2'])
    for av in artifact_versions:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))

    # 2. As a Data Scientist, I want to compare model performance in US-West and EU-Central regions.
    #    Action: Filter by `us-west-2`, `eu-central-1` tags.
    #    Scenario: Compare models deployed in US-West and EU-Central based on performance metrics.
    #    Benefit: Identifies performance variations between regions for optimization.

    print("\n\nArtifact Version Tag Use Case #2")
    print("As a Data Scientist, I want to compare model performance in US-West and EU-Central regions.")
    print("---------------------------------------\n")
    artifact_versions_us_west_2 = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['production', 'us-west-2'])
    artifact_versions_eu_central_1 = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['production', 'eu-central-1'])
    for av in artifact_versions_us_west_2:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))
    for av in artifact_versions_eu_central_1:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))

    # 3. As an ML Ops Engineer, I need to check the production model in EU-Central to troubleshoot an issue.
    #    Action: Filter by `production`, `eu-central-1`, and model_version tags.
    #    Scenario: Quickly find the production model version deployed in EU-Central for debugging.
    #    Benefit: Saves time and ensures the correct model is under investigation.

    print("\n\nArtifact Version Tag Use Case #3")
    print("As an ML Ops Engineer, I need to check the production model in EU-Central to troubleshoot an issue.")
    print("---------------------------------------\n")
    artifact_versions = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['production', 'eu-central-1'])
    for av in artifact_versions:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))

    # 4. As a Product Manager, I want to review all candidate models in ME-Central for possible promotion.
    #    Action: Filter by `candidate`, `me-central-1`, and base_model tags.
    #    Scenario: Gather models tagged as candidates for the ME-Central region and evaluate for production.
    #    Benefit: Streamlines the process of selecting models for promotion.

    print("\n\nArtifact Version Tag Use Case #4")
    print("As a Product Manager, I want to review all candidate models in ME-Central for possible promotion.")
    print("---------------------------------------\n")
    artifact_versions = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['candidate', 'me-central-1'])
    for av in artifact_versions:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))

    # 5. As a Compliance Officer, I need to audit archived models in US-West and ME-Central.
    #    Action: Filter by `archived`, `us-west-2`, and `me-central-1` tags.
    #    Scenario: Find older models in these regions to ensure they meet compliance requirements.
    #    Benefit: Efficiently audits the models without needing manual searches.

    print("\n\nArtifact Version Tag Use Case #5")
    print("As a Compliance Officer, I need to audit archived models in US-West and ME-Central.")
    print("---------------------------------------\n")
    artifact_versions_us_west_2 = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['archived', 'us-west-2'])
    artifact_versions_me_central_1 = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['archived', 'me-central-1'])
    for av in artifact_versions_us_west_2:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))
    for av in artifact_versions_me_central_1:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))

    # 6. As an ML Engineer, I want to evaluate models using the `Claude 3_5 Sonnet` base model in AP-East and EU-Central regions.
    #    Action: Filter by `Claude 3_5 Sonnet`, `ap-east-1`, and `eu-central-1` tags.
    #    Review and compare models fine-tuned on `Claude 3_5 Sonnet` across these regions.
    #    Benefit: Helps assess and improve performance for models based on the `Claude 3_5 Sonnet` base.

    print("\n\nArtifact Version Tag Use Case #6")
    print("As an ML Engineer, I want to evaluate models using the `Claude 3_5 Sonnet` base model in AP-East and EU-Central regions.")
    print("---------------------------------------\n")
    artifact_versions_ap_east_1 = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['Claude 3_5 Sonnet', 'ap-east-1'])
    artifact_versions_eu_central_1 = api.artifacts(type_name="model", name="wandb-registry-model/Artifact Demo Models", tags=['Claude 3_5 Sonnet', 'eu-central-1'])
    for av in artifact_versions_ap_east_1:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))
    for av in artifact_versions_eu_central_1:
      print("Artifact Version: " + str(av.name) + "\nTags: " + str(av.tags) + "\nVersion: " + str(av.version) + "\nMetadata: " + str(av.metadata))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Artifact version filtering results

    As you can see from the output of the script, it is possible to use one or more tags to retrieve one or more artifact versions from your registry collection.

    It is also possible to search by collection names, tags, and version tags from within the UI. Just use the search bar inside of any registry to find the artifacts that you need. The following screenshot shows the results for a search on "ap-east-1" in our Model registry.
    <br><br>

    <img src="https://rratshin-images.s3.us-west-2.amazonaws.com/colab_artifact_ui_search3.png" width="800" alt="Weights & Biases" border="1" />
    """)
    return


if __name__ == "__main__":
    app.run()
