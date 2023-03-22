from pathlib import Path

from nb_helpers.actions import upload_modified_nbs

scripts_folder = Path(__file__).resolve().parent

upload_modified_nbs("wandb", "examples", pr_message=scripts_folder/"pr_message.md")