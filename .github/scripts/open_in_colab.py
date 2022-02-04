from fastcore.all import *
from ghapi.all import *
from nb_helpers.utils import git_local_repo, is_nb


def create_comment():
    "On PR post a comment with links to open in colab for each changed nb"

    api = GhApi(owner="wandb", repo="nb_helpers", token=github_token())
    payload = context_github.event

    # if "workflow" in payload:
    #     issue = 1
    # else:
    #     if payload.action != "opened":
    #         return
    
    issue = payload.number
    print(f' >> {payload}\n')

    print(f' >> issue_number {issue}\n')

    pr = api.pulls.get(issue)
    github_repo, branch = pr.head.repo.full_name, pr.head.ref

    pr_files = [Path(f.filename) for f in api.pulls.list_files(issue)]

    # filter nbs
    nb_files = [f for f in pr_files if is_nb(f)]

    def _get_colab_url2md(fname: Path, github_repo=github_repo, branch=branch) -> str:
        "Create colab links in md"
        fname = fname.relative_to(git_local_repo(fname))
        colab_url = f"https://colab.research.google.com/github/{github_repo}/blob/{branch}/{str(fname)}"
        return f"- [{fname}]({colab_url})\n"

    def _create_comment_body(nb_files) -> str:
        "Creates a MD list of fnames with links to colab"
        title = "The following colabs where changed in this PR:\n"
        colab_links = tuple(_get_colab_url2md(f) for f in nb_files)
        body = tuplify(title) + colab_links
        return "".join(body)

    if len(nb_files) > 0:
        body = _create_comment_body(nb_files)
        print(f">> Creating comment on PR #{issue}")
        api.issues.create_comment(issue_number=issue, body=body)

create_comment()
