from fastcore.all import *
from ghapi.all import *
from nb_helpers.colab import get_colab_url
from nb_helpers.utils import git_current_branch, is_nb

def create_comment():
    "On PR post a comment with links to open in colab for each changed nb"

    api = GhApi(owner='wandb', repo='examples', token=github_token())
    payload = context_github.event
    if 'workflow' in payload: issue = 1
    else:
        if payload.action != 'opened': return
        issue = payload.number
    
    pr_files = [Path(f.filename) for f in api.pulls.list_files(issue)]
    
    #filter nbs
    nb_files = [f for f in pr_files if is_nb(f)]

    def _get_colab_url2md(fname: Path) -> str:
        "Create colab links in md"
        url = get_colab_url(fname, git_current_branch(fname))
        return f"- [{fname}]({url})\n"

    def _create_comment_body(nb_files: list[Path]) -> str:
        "Creates a MD list of fnames with links to colab"
        title = "The following colabs where changed in this PR:\n"
        colab_links = tuple(_get_colab_url2md(f) for f in nb_files)
        body = tuplify(title) + colab_links
        return "".join(body)
        
    if len(nb_files)>0:
        body = _create_comment_body(nb_files)
        print(f">> Creating comment on PR #{issue}")
        api.issues.create_comment(issue_number=issue, body=body)
    
create_comment()