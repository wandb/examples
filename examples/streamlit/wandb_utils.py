import wandb
from dotenv import load_dotenv, find_dotenv
import pandas as pd


load_dotenv(find_dotenv())

api = wandb.Api()


def get_run_iframe(run_path, height=720):
    run = api.from_path(run_path)
    return run.to_html(height=height, hidden=False)


def get_projects(entity, height=720):
    projects = {}
    for project in api.projects(entity):
        projects[project.name] = project.to_html(height=height)
    return projects


def get_runs(entity, project):

    runs = api.runs(f"{entity}/{project}")

    id_list, state_list, summary_list, config_list, name_list = [], [], [], [], []
    for run in runs:
        id_list.append(run.id)
        state_list.append(run.state)
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "id": id_list,
        "state": state_list,
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })

    return runs_df


def get_wandb_demo_artifact(project_path):

    api = wandb.Api()
    artifact = api.artifact(
        f'{project_path}/demo_artifacts:latest', type='demo')
    artifact_dir = artifact.download()

    return artifact_dir


def save_example_html(dest_path="example.html"):

    with open(dest_path, 'w') as f:
        content = """
        <!DOCTYPE html>
        <html>
        <body>

        <h1 style="color:blue">Example HTML</h1>
        <p style="color:blue">This HTML is logged and tracked as a W&B Artifact. 
        You can log anything as a W&B Artifact including models & datasets and they will be tracked and versioned anytime you update them.
        </p>

        </body>
        </html>
        """
        f.write(content)


def log_example_html_to_wandb():
    run = wandb.init(project="Log-Example-HTML")
    dest_path = "example.html"
    save_example_html(dest_path=dest_path)

    demo_artifacts = wandb.Artifact("demo_artifacts", type="demo")
    demo_html = wandb.Html(dest_path)
    # Here you can add your example artifacts
    demo_artifacts.add(demo_html, "demo_html")
    run.log_artifact(demo_artifacts)
    run.finish()
    return None


if __name__ == "__main__":
    log_example_html_to_wandb()
