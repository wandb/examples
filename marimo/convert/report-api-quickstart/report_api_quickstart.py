# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "pandas>=2.0",
#     "pillow>=10.0",
#     "requests>=2.31",
#     "wandb>=0.18",
#     "wandb-workspaces>=0.1",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="W&B Report API Quickstart")

with app.setup:
    import math
    import random
    import string
    from io import BytesIO
    from itertools import product

    import marimo as mo
    import pandas as pd
    import requests
    import wandb
    import wandb_workspaces.reports.v2 as wr
    from PIL import Image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # W&B Report API

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/report-api-quickstart/report_api_quickstart.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    Programmatically create, manage, and customize Reports by defining configurations, panel layouts, and runsets with the wandb-workspaces W&B library. Load and modify Reports with URLs, filter and group runs using expressions, and customize run appearances using Report templates.

    [wandb-workspaces](https://github.com/wandb/wandb-workspaces) is a Python library for programmatically creating and customizing W&B Workspaces and Reports.

    In this tutorial you will see how to use wandb-workspaces to create and customize W&B Reports.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    The notebook dependencies install the W&B SDK and `wandb-workspaces` automatically. Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, a team entity. You can leave the key blank when this environment already has credentials, such as `WANDB_API_KEY` configured in the marimo Secrets panel. If you are new to W&B, [create a free account](https://wandb.ai/signup).

    Connecting only authenticates and resolves your entity. It does not create runs or Reports. `wandb-workspaces` is currently in Public Preview, and Report objects are uploaded only when the tutorial explicitly calls `report.save()` after you click a creation button.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use configured credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="W&B entity or team (optional)",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    _project_input = mo.ui.text(
        value="report-api-quickstart",
        label="Report project",
        full_width=True,
    )
    _lineage_project_input = mo.ui.text(
        value="lineage-example",
        label="Artifact lineage project",
        full_width=True,
    )
    wandb_login_form = (
        mo.md(
            "{api_key}\n\n{entity}\n\n{project}\n\n{lineage_project}"
        )
        .batch(
            api_key=_api_key_input,
            entity=_entity_input,
            project=_project_input,
            lineage_project=_lineage_project_input,
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Connect to W&B before using the Report API examples."),
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _submitted_entity = wandb_login_form.value["entity"].strip()
    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
        )
        _login_error = None
    except wandb.errors.Error as _error:
        _login_ok = False
        _login_error = str(_error)

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key and "
                f"try again.\n\nW&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    ENTITY = _submitted_entity or wandb.Api().default_entity
    mo.stop(
        not ENTITY,
        mo.callout(
            mo.md(
                "W&B did not return a default entity. Enter a team entity in "
                "the form above and reconnect."
            ),
            kind="danger",
        ),
    )
    PROJECT = wandb_login_form.value["project"].strip() or "report-api-quickstart"
    LINEAGE_PROJECT = (
        wandb_login_form.value["lineage_project"].strip() or "lineage-example"
    )
    mo.callout(
        mo.md(
            f"Connected to W&B as `{ENTITY}`. Report examples will use "
            f"`{PROJECT}` and lineage examples will use `{LINEAGE_PROJECT}`."
        ),
        kind="success",
    )
    return ENTITY, LINEAGE_PROJECT, PROJECT


@app.function
def get_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


@app.function
def log_dummy_data(entity, project):
    run_names = [
        "adventurous-aardvark-1",
        "bountiful-badger-2",
        "clairvoyant-chipmunk-3",
        "dastardly-duck-4",
        "eloquent-elephant-5",
        "flippant-flamingo-6",
        "giddy-giraffe-7",
        "haughty-hippo-8",
        "ignorant-iguana-9",
        "jolly-jackal-10",
        "kind-koala-11",
        "laughing-lemur-12",
        "manic-mandrill-13",
        "neighbourly-narwhal-14",
        "oblivious-octopus-15",
        "philistine-platypus-16",
        "quant-quail-17",
        "rowdy-rhino-18",
        "solid-snake-19",
        "timid-tarantula-20",
        "understanding-unicorn-21",
        "voracious-vulture-22",
        "wu-tang-23",
        "xenic-xerneas-24",
        "yielding-yveltal-25",
        "zooming-zygarde-26",
    ]

    opts = ["adam", "sgd"]
    encoders = ["resnet18", "resnet50"]
    learning_rates = [0.01]
    for (i, run_name), (opt, encoder, lr) in zip(
        enumerate(run_names), product(opts, encoders, learning_rates)
    ):
        config = {
            "optimizer": opt,
            "encoder": encoder,
            "learning_rate": lr,
            "momentum": 0.1 * random.random(),
        }
        displacement1 = random.random() * 2
        displacement2 = random.random() * 4
        with wandb.init(
            entity=entity, project=project, config=config, name=run_name
        ) as run:
            for step in range(1000):
                run.log(
                    {
                        "acc": 0.1
                        + 0.4
                        * (
                            math.log(1 + step + random.random())
                            + random.random() * run.config.learning_rate
                            + random.random()
                            + displacement1
                            + random.random() * run.config.momentum
                        ),
                        "val_acc": 0.1
                        + 0.4
                        * (
                            math.log(1 + step + random.random())
                            + random.random() * run.config.learning_rate
                            - random.random()
                            + displacement1
                        ),
                        "loss": 0.1
                        + 0.08
                        * (
                            3.5
                            - math.log(1 + step + random.random())
                            + random.random() * run.config.momentum
                            + random.random()
                            + displacement2
                        ),
                        "val_loss": 0.1
                        + 0.04
                        * (
                            4.5
                            - math.log(1 + step + random.random())
                            + random.random() * run.config.learning_rate
                            - random.random()
                            + displacement2
                        ),
                    }
                )

    with wandb.init(
        entity=entity, project=project, config=config, name=run_names[i + 1]
    ) as run:
        img = get_image(
            "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
        )
        image = wandb.Image(img)
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.2, 2.3, 3.4, 4.5],
                "str": ["a", "b", "c", "d"],
                "img": [image] * 4,
            }
        )
        run.log({"img": image, "my-table": df})


@app.class_definition
class Step:
    def __init__(self, j, r, u, o, at=None):
        self.job_type = j
        self.runs = r
        self.uses_per_run = u
        self.outputs_per_run = o
        self.artifact_type = at if at is not None else "model"
        self.artifacts = []


@app.function
def create_artifact(name: str, type: str, content: str):
    art = wandb.Artifact(name, type)
    with art.new_file("test-name", mode="w") as f:
        f.write(content)

    img = get_image(
        "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
    )
    image = wandb.Image(img)
    df = pd.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float": [1.2, 2.3, 3.4, 4.5],
            "str": ["a", "b", "c", "d"],
            "img": [image] * 4,
        }
    )
    art.add(wandb.Table(dataframe=df), "dataframe")
    return art


@app.function
def log_dummy_lineage(entity, lineage_project):
    pipeline = [
        Step("dataset-generator", 1, 0, 3, "dataset"),
        Step("trainer", 4, (1, 2), 3),
        Step("evaluator", 2, 1, 3),
        Step("ensemble", 1, 1, 1),
    ]
    for i, step in enumerate(pipeline):
        for _ in range(step.runs):
            with wandb.init(
                entity=entity,
                project=lineage_project,
                job_type=step.job_type,
            ) as run:
                # use
                uses = step.uses_per_run
                if type(uses) == tuple:
                    uses = random.choice(list(uses))

                if i > 0:
                    prev_step = pipeline[i - 1]
                    input_artifacts = random.sample(prev_step.artifacts, uses)
                    for a in input_artifacts:
                        run.use_artifact(a)
                # log output artifacts
                for j in range(step.outputs_per_run):
                    # name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
                    name = f"{step.artifact_type}-{j}"
                    content = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=12)
                    )
                    art = create_artifact(name, step.artifact_type, content)
                    run.log_artifact(art)
                    art.wait()

                    # save in pipeline
                    step.artifacts.append(art)


@app.cell(hide_code=True)
def _(ENTITY, LINEAGE_PROJECT, PROJECT):
    _log_dummy_runs = mo.ui.checkbox(
        value=True,
        label="Log demo runs and Artifact lineage",
    )
    demo_data_form = (
        mo.md("{log_dummy_runs}")
        .batch(log_dummy_runs=_log_dummy_runs)
        .form(
            submit_button_label="Create 13 runs + Artifacts / skip unchecked",
            bordered=True,
        )
    )
    mo.vstack(
        [
            mo.md(
                "The original tutorial can create five metric/media runs in "
                f"`{ENTITY}/{PROJECT}` and eight lineage runs plus Artifacts in "
                f"`{ENTITY}/{LINEAGE_PROJECT}`. Keep the box checked to create "
                "that remote demo data, or clear it to skip the optional setup. "
                "Changing the checkbox does nothing until you submit the form."
            ),
            demo_data_form,
        ]
    )
    return (demo_data_form,)


@app.cell(hide_code=True)
def _(ENTITY, LINEAGE_PROJECT, PROJECT, demo_data_form):
    mo.stop(
        demo_data_form.value is None,
        mo.callout(
            mo.md("Submit the demo-data choice above to continue."),
            kind="info",
        ),
    )
    LOG_DUMMY_RUNS = bool(demo_data_form.value["log_dummy_runs"])
    if LOG_DUMMY_RUNS:
        log_dummy_data(ENTITY, PROJECT)
        log_dummy_lineage(ENTITY, LINEAGE_PROJECT)
        _demo_status = "Created the optional metric/media runs and Artifact lineage."
    else:
        _demo_status = "Skipped the optional demo runs and Artifact lineage."
    mo.callout(mo.md(_demo_status), kind="success")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Quickstart! <a id='quickstart'></a>
    """)
    return


@app.cell(hide_code=True)
def _(PROJECT):
    create_quickstart_report = mo.ui.run_button(
        label="Create and update the Quickstart Report"
    )
    mo.vstack(
        [
            mo.md(
                f"This action creates one Report in `{PROJECT}`, then saves "
                "each of the changes shown in the next four code cells. A new "
                "button click intentionally performs the sequence again."
            ),
            create_quickstart_report,
        ]
    )
    return (create_quickstart_report,)


@app.cell(hide_code=True)
def _(ENTITY, PROJECT, create_quickstart_report):
    mo.stop(
        not create_quickstart_report.value,
        mo.callout(
            mo.md("Click the button above when you are ready to create the Report."),
            kind="info",
        ),
    )
    quickstart_target = {"entity": ENTITY, "project": PROJECT}
    return (quickstart_target,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Create, save, and load reports
    - NOTE: Reports are not saved automatically to reduce clutter.  Explicitly save the report by calling `report.save()`
    """)
    return


@app.cell
def _(quickstart_target):
    report = wr.Report(
        entity=quickstart_target["entity"],
        project=quickstart_target["project"],
        title="Quickstart Report",
        description="That was easy!",
    )                                 # Create
    report.save()                     # Save
    wr.Report.from_url(report.url)    # Load
    return (report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Add content via blocks
    - Use blocks to add content like text, images, code, and more
    - See `wr.blocks` for all available blocks
    """)
    return


@app.cell
def _(report):
    report.blocks = [
        wr.TableOfContents(),
        wr.H1("Text and images example"),
        wr.P("Lorem ipsum dolor sit amet. Aut laborum perspiciatis sit odit omnis aut aliquam voluptatibus ut rerum molestiae sed assumenda nulla ut minus illo sit sunt explicabo? Sed quia architecto est voluptatem magni sit molestiae dolores. Non animi repellendus ea enim internos et iste itaque quo labore mollitia aut omnis totam."),
        wr.Image("https://api.wandb.ai/files/telidavies/images/projects/831572/8ad61fd1.png", caption="Craiyon generated images"),
        wr.P("Et voluptatem galisum quo facilis sequi quo suscipit sunt sed iste iure! Est voluptas adipisci et doloribus commodi ab tempore numquam qui tempora adipisci. Eum sapiente cupiditate ut natus aliquid sit dolor consequatur?"),
    ]
    report.save()
    report_with_content = report
    return (report_with_content,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Add charts and more via Panel Grid
    - `PanelGrid` is a special type of block that holds `runsets` and `panels`
      - `runsets` organize data logged to W&B
      - `panels` visualize runset data.  For a full set of panels, see `wr.panels`
    """)
    return


@app.cell
def _(ENTITY, PROJECT, report_with_content):
    pg = wr.PanelGrid(
        runsets=[
            wr.Runset(ENTITY, PROJECT, "First Run Set"),
            wr.Runset(ENTITY, PROJECT, "Elephants Only!", query="elephant"),
        ],
        panels=[
            wr.LinePlot(x="Step", y=["val_acc"], smoothing_factor=0.8),
            wr.BarPlot(metrics=["acc"]),
            wr.MediaBrowser(media_keys=["img"], num_columns=1),
            wr.RunComparer(diff_only="split", layout={"w": 24, "h": 9}),
        ],
    )

    report_with_content.blocks = report_with_content.blocks[:1] + [wr.H1("Panel Grid Example"), pg] + report_with_content.blocks[1:]
    report_with_content.save()
    report_with_panel_grid = report_with_content
    return pg, report_with_panel_grid


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Add data lineage with Artifact blocks
    - There are equivalent weave panels as well
    """)
    return


@app.cell
def _(ENTITY, LINEAGE_PROJECT, report_with_panel_grid):
    artifact_lineage = wr.WeaveBlockArtifact(
        entity=ENTITY,
        project=LINEAGE_PROJECT,
        artifact="model-1",
        tab="lineage",
    )

    report_with_panel_grid.blocks = report_with_panel_grid.blocks[:1] + [wr.H1("Artifact lineage example"), artifact_lineage] + report_with_panel_grid.blocks[1:]
    report_with_panel_grid.save()
    report_with_lineage = report_with_panel_grid
    return (report_with_lineage,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Customize run colors
    - Pass in a `dict[run_name, color]`
    """)
    return


@app.cell
def _(pg, report_with_lineage):
    pg.custom_run_colors = {
        "adventurous-aardvark-1": "#e84118",
        "bountiful-badger-2":     "#fbc531",
        "clairvoyant-chipmunk-3": "#4cd137",
        "dastardly-duck-4":       "#00a8ff",
        "eloquent-elephant-5":    "#9c88ff",
    }
    report_with_lineage.save()
    quickstart_report = report_with_lineage
    quickstart_report_complete = quickstart_report.url
    return quickstart_report, quickstart_report_complete


@app.cell(hide_code=True)
def _(quickstart_report_complete):
    mo.callout(
        mo.md(f"Quickstart complete: [open the saved Report]({quickstart_report_complete})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## FAQ <a id='faq'></a>
    """)
    return


@app.cell(hide_code=True)
def _(PROJECT, quickstart_report_complete):
    create_faq_reports = mo.ui.run_button(label="Create the FAQ example Reports")
    mo.vstack(
        [
            mo.md(
                f"This action clones the [Quickstart Report]({quickstart_report_complete}) and creates seven more "
                f"Reports in `{PROJECT}` for the examples below. Each click "
                "intentionally creates a fresh set of eight Reports."
            ),
            create_faq_reports,
        ]
    )
    return (create_faq_reports,)


@app.cell(hide_code=True)
def _(create_faq_reports, quickstart_report):
    mo.stop(
        not create_faq_reports.value,
        mo.callout(
            mo.md("Click the button above when you are ready to create the FAQ Reports."),
            kind="info",
        ),
    )
    faq_source_report = quickstart_report
    return (faq_source_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### My report is too wide/narrow
    - Change the report's width to the right size for you.
    """)
    return


@app.cell
def _(faq_source_report):
    report2 = faq_source_report.save(clone=True)
    report2.width = "fluid"
    report2.save()
    faq_report_urls = [report2.url]
    return (faq_report_urls,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How do I resize panels?
    - Pass a `dict[dim, int]` to `panel.layout`
    - `dim` is a dimension, which can be `x`, `y` (the coordiantes of the top left corner) `w`, `h` (the size of the panel)
    - You can pass any or all dimensions at once
    - The space between two dots in a panel grid is 2.
    """)
    return


@app.cell
def _(ENTITY, PROJECT, faq_report_urls):
    resize_report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Resizing panels",
        description="Look at this wide parallel coordinates plot!",
        blocks=[
            wr.PanelGrid(
                panels=[
                    wr.ParallelCoordinatesPlot(
                        columns=[
                            wr.ParallelCoordinatesPlotColumn(metric="Step"),
                            wr.ParallelCoordinatesPlotColumn(metric="c::model"),
                            wr.ParallelCoordinatesPlotColumn(metric="c::optimizer"),
                            wr.ParallelCoordinatesPlotColumn(metric="Step"),
                            wr.ParallelCoordinatesPlotColumn(metric="val_acc"),
                            wr.ParallelCoordinatesPlotColumn(metric="val_loss"),
                        ],
                        layout=wr.Layout(w=24, h=9)  # Adjusting the layout for the plot size
                    ),
                ]
            )
        ],
    )
    resize_report.save()
    faq_report_urls_after_resize = [*faq_report_urls, resize_report.url]
    return (faq_report_urls_after_resize,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What blocks are available?
    - See `wr.blocks` for a list of blocks.
    - In an IDE or notebook, you can also do `wr.blocks.<tab>` to get autocomplete.
    """)
    return


@app.cell
def _(ENTITY, LINEAGE_PROJECT, PROJECT, faq_report_urls_after_resize):
    block_gallery_report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="W&B Block Gallery",
        description="Check out all of the blocks available in W&B",
        blocks=[
            wr.H1(text="Heading 1"),
            wr.P(text="Normal paragraph"),
            wr.H2(text="Heading 2"),
            wr.P(
                text=[
                    "here is some text, followed by",
                    wr.InlineCode(text="select * from code in line"),
                    "and then latex",
                    wr.InlineLatex(text="e=mc^2"),
                ]
            ),
            wr.H3(text="Heading 3"),
            wr.CodeBlock(
                code="this:\n- is\n- a\ncool:\n- yaml\n- file",
                language="yaml",
            ),
            wr.WeaveBlockSummaryTable(entity=ENTITY, project=PROJECT, table_name="my-table"),
            wr.WeaveBlockArtifact(entity=ENTITY, project=LINEAGE_PROJECT, artifact="model-1", tab="lineage"),
            wr.WeaveBlockArtifactVersionedFile(entity=ENTITY, project=LINEAGE_PROJECT, artifact="model-1", version="v0", file="dataframe.table.json"),
            wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$"),
            wr.LatexBlock(text="\\gamma^2+\\theta^2=\\omega^2\n\\\\ a^2 + b^2 = c^2"),
            wr.Image(url="https://api.wandb.ai/files/megatruong/images/projects/918598/350382db.gif", caption="It's a me, Pikachu"),
            wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
            wr.OrderedList(items=["Ordered 1", "Ordered 2"]),
            wr.CheckedList(items=[wr.CheckedListItem(text="Unchecked", checked=False), wr.CheckedListItem(text="Checked", checked=True)]),
            wr.BlockQuote(text="Block Quote 1\nBlock Quote 2\nBlock Quote 3"),
            wr.CalloutBlock(text="Callout 1\nCallout 2\nCallout 3"),
            wr.HorizontalRule(),
            wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
        ],
    )
    block_gallery_report.save()
    faq_report_urls_after_blocks = [
        *faq_report_urls_after_resize,
        block_gallery_report.url,
    ]
    return (faq_report_urls_after_blocks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### What panels are available?
    - See `wr.panels` for a list of panels
    - In an IDE or notebook, you can also do `wr.panels.<tab>` to get autocomplete.
    - Panels have a lot of settings.  Inspect the panel to see what you can do!
    """)
    return


@app.cell
def _(ENTITY, LINEAGE_PROJECT, PROJECT, faq_report_urls_after_blocks):
    panel_gallery_report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="W&B Panel Gallery",
        description="Check out all of the panels available in W&B",
        width="fluid",
        blocks=[
            wr.PanelGrid(
                runsets=[wr.Runset(project=LINEAGE_PROJECT), wr.Runset()],
                panels=[
                    wr.MediaBrowser(media_keys=["img"]),
                    wr.MarkdownPanel(markdown="Hello *italic* **bold** $e=mc^2$ `something`"),

                    # LinePlot with various settings enabled
                    wr.LinePlot(
                        title="Validation Accuracy over Time",
                        x="Step",
                        y=["val_acc"],
                        range_x=(0, 1000),
                        range_y=(1, 4),
                        log_x=True,
                        log_y=False,
                        title_x="Training steps",
                        title_y="Validation Accuracy",
                        ignore_outliers=True,
                        groupby="encoder",
                        groupby_aggfunc="mean",
                        groupby_rangefunc="minmax",
                        smoothing_factor=0.5,
                        smoothing_type="gaussian",
                        smoothing_show_original=True,
                        max_runs_to_show=10,
                        font_size="large",
                        legend_position="west",
                    ),
                    wr.ScatterPlot(title="Validation Accuracy vs. Validation Loss", x="val_acc", y="val_loss", log_x=False, log_y=False, running_ymin=True, running_ymean=True, running_ymax=True, font_size="small", regression=True),
                    wr.BarPlot(title="Validation Loss by Encoder", metrics=["val_loss"], orientation="h", range_x=(0, 0.11), title_x="Validation Loss", groupby="encoder", groupby_aggfunc="median", groupby_rangefunc="stddev", max_runs_to_show=20, max_bars_to_show=3, font_size="auto"),
                    wr.ScalarChart(title="Maximum Number of Steps", metric="Step", groupby_aggfunc="max", groupby_rangefunc="stderr", font_size="large"),
                    wr.CodeComparer(diff="split"),
                    wr.ParallelCoordinatesPlot(columns=[wr.ParallelCoordinatesPlotColumn("Step"), wr.ParallelCoordinatesPlotColumn("c::model"), wr.ParallelCoordinatesPlotColumn("c::optimizer"), wr.ParallelCoordinatesPlotColumn("val_acc"), wr.ParallelCoordinatesPlotColumn("val_loss")]),
                    wr.ParameterImportancePlot(with_respect_to="val_loss"),
                    wr.RunComparer(diff_only=True),
                    wr.CustomChart(query={"summary": ["val_loss", "val_acc"]}, chart_name="wandb/scatter/v0", chart_fields={"x": "val_loss", "y": "val_acc"}),
                ],
            ),
            # Add WeaveBlock types directly to the blocks list
            wr.WeaveBlockSummaryTable(entity="your_entity", project="your_project", table_name="my-table"),
            wr.WeaveBlockArtifact(entity="your_entity", project="your_project", artifact="model-1", tab="lineage"),
            wr.WeaveBlockArtifactVersionedFile(entity="your_entity", project="your_project", artifact="model-1", version="v0", file="dataframe.table.json"),
        ],
    )
    panel_gallery_report.save()
    faq_report_urls_after_panels = [
        *faq_report_urls_after_blocks,
        panel_gallery_report.url,
    ]
    return (faq_report_urls_after_panels,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### How can I link related reports together?
    - Suppose have have two reports like below:
    """)
    return


@app.cell
def _(ENTITY, PROJECT, faq_report_urls_after_panels):
    report1 = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Report 1",
        description="Great content coming from Report 1",
        blocks=[
            wr.H1(text="Heading from Report 1"),
            wr.P(text="Lorem ipsum dolor sit amet. Aut fuga minus nam vero saepeA aperiam eum omnis dolorum et ducimus tempore aut illum quis aut alias vero. Sed explicabo illum est eius quianon vitae sed voluptatem incidunt. Vel architecto assumenda Ad voluptatem quo dicta provident et velit officia. Aut galisum inventoreSed dolore a illum adipisci a aliquam quidem sit corporis quia cum magnam similique."),
            wr.PanelGrid(
                panels=[
                    wr.LinePlot(title="Episodic Return", x="global_step", y=["charts/episodic_return"], smoothing_factor=0.85, groupby_aggfunc="mean", groupby_rangefunc="minmax", layout=wr.Layout(x=0, y=0, w=12, h=8)),
                    wr.MediaBrowser(media_keys=["videos"], num_columns=4, layout=wr.Layout(w=12, h=8)),
                ],
                runsets=[wr.Runset(entity="openrlbenchmark", project="cleanrl", query="bigfish", groupby=["env_id", "exp_name"])],
                custom_run_colors={
                    wr.RunsetGroup(runset_name="Run set", keys=(wr.RunsetGroupKey(key="bigfish", value="ppg_procgen"),)): "#2980b9",
                    wr.RunsetGroup(runset_name="Run set", keys=(wr.RunsetGroupKey(key="bigfish", value="ppo_procgen"),)): "#e74c3c",
                },
            ),
        ],
    )
    report1.save()

    report2_1 = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Report 2",
        description="Great content coming from Report 2",
        blocks=[
            wr.H1(text="Heading from Report 2"),
            wr.P(text="Est quod ducimus ut distinctio corruptiid optio qui cupiditate quibusdam ea corporis modi. Eum architecto vero sed error dignissimosEa repudiandae a recusandae sint ut sint molestiae ea pariatur quae. In pariatur voluptas ad facere neque 33 suscipit et odit nostrum ut internos molestiae est modi enim. Et rerum inventoreAut internos et dolores delectus aut Quis sunt sed nostrum magnam ab dolores dicta."),
            wr.PanelGrid(
                panels=[
                    wr.LinePlot(title="SPS", x="global_step", y=["charts/SPS"]),
                    wr.LinePlot(title="Episodic Length", x="global_step", y=["charts/episodic_length"]),
                    wr.LinePlot(title="Episodic Return", x="global_step", y=["charts/episodic_return"]),
                ],
                runsets=[
                    wr.Runset(entity="openrlbenchmark", project="cleanrl", name="DQN", groupby=["exp_name"]),
                    wr.Runset(entity="openrlbenchmark", project="cleanrl", name="SAC-discrete 0.8", groupby=["exp_name"]),
                    wr.Runset(entity="openrlbenchmark", project="cleanrl", name="SAC-discrete 0.88", groupby=["exp_name"]),
                ],
                custom_run_colors={
                    wr.RunsetGroup(runset_name="DQN", keys=(wr.RunsetGroupKey(key="dqn_atari", value="exp_name"),)): "#e84118",
                    wr.RunsetGroup(runset_name="SAC-discrete 0.8", keys=(wr.RunsetGroupKey(key="sac_atari", value="exp_name"),)): "#fbc531",
                    wr.RunsetGroup(runset_name="SAC-discrete 0.88", keys=(wr.RunsetGroupKey(key="sac_atari", value="exp_name"),)): "#00a8ff",
                },
            ),
        ],
    )
    report2_1.save()
    faq_report_urls_after_linked_pair = [
        *faq_report_urls_after_panels,
        report1.url,
        report2_1.url,
    ]
    return faq_report_urls_after_linked_pair, report1, report2_1


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Combine blocks into a new report
    """)
    return


@app.cell
def _(ENTITY, PROJECT, faq_report_urls_after_linked_pair):
    report_4 = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Report with links",
        description="Use `wr.Link(text, url)` to add links inside normal text, or use normal markdown syntax in a MarkdownBlock",
        blocks=[
            wr.H1("This is a normal heading"),
            wr.P("And here is some normal text"),

            wr.H1(["This is a heading ", wr.Link("with a link!", url="https://wandb.ai/")]),
            wr.P(["Most text formats support ", wr.Link("adding links", url="https://wandb.ai/")]),

            wr.MarkdownBlock("""You can also use markdown syntax for [links](https://wandb.ai/)"""),
        ],
    )
    report_4.save()
    faq_report_urls_after_links = [
        *faq_report_urls_after_linked_pair,
        report_4.url,
    ]
    return (faq_report_urls_after_links,)


@app.cell
def _(ENTITY, PROJECT, faq_report_urls_after_links, report1, report2_1):
    report3 = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="Combined blocks report",
        description="This report combines blocks from both Report 1 and Report 2",
        blocks=[*report1.blocks, *report2_1.blocks],
    )
    report3.save()
    faq_reports_complete = [*faq_report_urls_after_links, report3.url]
    return (faq_reports_complete,)


@app.cell(hide_code=True)
def _(faq_reports_complete):
    mo.callout(
        mo.md(
            f"Created {len(faq_reports_complete)} FAQ example Reports. "
            f"[Open the combined Report]({faq_reports_complete[-1]})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### I tried mutating an object in list but it didn't work!
    tl;dr: It should always work if you assign a value to the attribute instead of mutating.  If you really need to mutate, do it before assignment.

    ---

    This can happen in a few places that contain lists of wandb objects, e.g.:
    - `report.blocks`
    - `panel_grid.panels`
    - `panel_grid.runsets`

    The current `wandb-workspaces` release keeps the assigned block object by reference, so the in-place mutation below is visible through `report.blocks`. Explicitly reassigning the block list still makes the intended update unambiguous before a later `save()`.

    The following object stays local because none of these cells calls `save()`.
    """)
    return


@app.cell
def _(ENTITY, PROJECT):
    report_5 = wr.Report(entity=ENTITY, project=PROJECT)
    return (report_5,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Good: Assign `b`
    """)
    return


@app.cell
def _(report_5):
    b = wr.H1(text=["Hello", " World!"])
    report_5.blocks = [b]
    assert b.text == ["Hello", " World!"]
    assert report_5.blocks[0].text == ["Hello", " World!"]
    initial_assignment_done = True
    return (b,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Current behavior: Mutate `b` without reassigning
    """)
    return


@app.cell
def _(b, report_5):
    b.text = ["Something", " New"]
    assert b.text == ["Something", " New"]
    assert report_5.blocks[0].text == ["Something", " New"]

    # Older releases could leave the previously assigned value here.
    mutation_example_done = True
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Recommended: Mutate `b` and then reassign it
    """)
    return


@app.cell
def _(b, report_5):
    report_5.blocks = [b]
    assert b.text == ["Something", " New"]
    assert report_5.blocks[0].text == ["Something", " New"]
    return


if __name__ == "__main__":
    app.run()
