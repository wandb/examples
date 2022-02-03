from fastcore.all import *
from ghapi.all import *
from nb_helpers.colab import get_colab_url
from nb_helpers.utils import git_current_branch

def create_comment():
    api = GhApi(owner='wandb', repo='examples', token=github_token())
    payload = context_github.event
    if 'workflow' in payload: issue = 1
    else:
        if payload.action != 'opened': return
        issue = payload.number
    
    pr_files = [f.filename for f in api.pulls.list_files(issue)]
    nb_files = [f for f in pr_files if ".ipynb" in f]

    def get_colab_md(fname):
        _fname = Path(fname).name
        url = get_colab_url(fname, git_current_branch(fname))
        return f"- [{_fname}]({url})\n"

    colab_links = tuple(get_colab_md(f) for f in nb_files)
    body = ("The following colabs where changed in this PR:\n",)+colab_links
    body = "".join(body)

    api.issues.create_comment(issue_number=issue, body=body)

create_comment()