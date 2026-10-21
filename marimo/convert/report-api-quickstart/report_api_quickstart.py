# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "pandas>=2.0",
#     "pillow>=10.0",
#     "wandb>=0.19.10",
#     "wandb-workspaces>=0.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Weights & Biases Report API Quickstart",
)

with app.setup:
    import math
    import random
    from itertools import product

    import marimo as mo
    import pandas as pd
    import wandb
    import wandb_workspaces.reports.v2 as wr
    from PIL import Image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-config-logo--dark {
      display: none;
    }

    body.dark .wandb-config-logo--light {
      display: none;
    }

    body.dark .wandb-config-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-config-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-config-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />

    # Weights & Biases Report API Quickstart

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/report-api-quickstart/report_api_quickstart.py/server)

    Use Workspaces and Reports API ([wandb-workspaces](https://github.com/wandb/wandb-workspaces)) to create and
    customize Weights & Biases Reports. A report combines narrative **blocks** with **panel grids**:
    runsets select experiment data, and panels visualize that data.

    This tutorial covers how to create, save, and load a report. Along the way you will learn how to add common UI elements such as text, media, and panels to your report.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `report-api-quickstart`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="report-api-quickstart", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    entity = _entity
    project = _project
    mo.md("✓ Connected to Weights & Biases.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Create data for the panels

    Before we create a report, let's log a psuedo training loop and some images so have content to add to our report.

    The following cell defines some run names and colors that we will use. The specifics are not important.
    """)
    return


@app.cell
def _():
    demo_run_colors = {}

    demo_rng = random.Random(0)
    demo_run_names = [
        "adventurous-aardvark-1", "bountiful-badger-2",
        "clairvoyant-chipmunk-3", "dastardly-duck-4", "eloquent-elephant-5",
    ]
    demo_colors = ["#e84118", "#fbc531", "#4cd137", "#00a8ff", "#9c88ff"]
    return demo_colors, demo_rng, demo_run_colors, demo_run_names


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell initializes three different loops:
    - Training loop
    - Log an image
    - Log an artifact (for lineage)
    """)
    return


@app.cell
def _(demo_colors, demo_rng, demo_run_colors, demo_run_names, entity, project):
    # Log psuedo training metrics such as accuracy and loss
    for demo_index, (optimizer_name, encoder_name) in enumerate(product(
        ["adam", "sgd"], ["resnet18", "resnet50"],
    )):
        with wandb.init(
            entity=entity, project=project, name=demo_run_names[demo_index],
            config={"optimizer": optimizer_name, "encoder": encoder_name,
                    "learning_rate": 0.01, "momentum": demo_rng.random() / 10},
        ) as metrics_run:
            for step in range(100):
                progress = 1 - math.exp(-step / 30)
                noise = demo_rng.uniform(-0.02, 0.02)
                metrics_run.log({
                    "acc": 0.2 + 0.75 * progress + noise,
                    "val_acc": 0.2 + 0.7 * progress + noise,
                    "loss": 1.5 * (1 - progress) + 0.1,
                    "val_loss": 1.7 * (1 - progress) + 0.15,
                })
            demo_run_colors[metrics_run.id] = demo_colors[demo_index]


    # Create a color gradient image
    demo_image = Image.new("RGB", (64, 64))

    demo_image.putdata([(4 * x, 4 * y, 128) for y in range(64) for x in range(64)])
    table_image = wandb.Image(demo_image, caption="Synthetic color gradient")

    demo_dataframe = pd.DataFrame({
        "int": [1, 2, 3, 4], "float": [1.2, 2.3, 3.4, 4.5],
        "str": ["a", "b", "c", "d"], "img": [table_image] * 4,
    })

    # Log color gradient image
    with wandb.init(
        entity=entity, project=project, name=demo_run_names[-1], job_type="media",
    ) as media_run:
        media_run.log({"img": table_image, "my-table": wandb.Table(dataframe=demo_dataframe)})
        demo_run_colors[media_run.id] = demo_colors[-1]

    pipeline = [
        ("dataset-generator", 1, 0, 3, "dataset"),
        ("trainer", 4, (1, 2), 3, "model"),
        ("evaluator", 2, 1, 3, "model"),
        ("ensemble", 1, 1, 1, "model"),
    ]
    previous_artifacts = []

    # Log artifact
    for job_type, run_count, uses_per_run, output_count, artifact_type in pipeline:
        stage_artifacts = []
        for lineage_index in range(run_count):
            with wandb.init(
                entity=entity, project=project, job_type=job_type,
                name=f"report-demo-{job_type}-{lineage_index + 1}",
            ) as lineage_run:
                input_count = demo_rng.choice(uses_per_run) if isinstance(uses_per_run, tuple) else uses_per_run
                for input_artifact in demo_rng.sample(previous_artifacts, input_count):
                    lineage_run.use_artifact(input_artifact)
                for output_index in range(output_count):
                    lineage_artifact = wandb.Artifact(f"{artifact_type}-{output_index}", type=artifact_type)
                    with lineage_artifact.new_file("stage.txt", mode="w") as stage_file:
                        stage_file.write(f"{job_type}/{lineage_index}/{output_index}/{demo_rng.random()}")
                    lineage_artifact.add(wandb.Table(dataframe=demo_dataframe), "dataframe")
                    uploaded_artifact = lineage_run.log_artifact(lineage_artifact)
                    uploaded_artifact.wait()
                    stage_artifacts.append(uploaded_artifact)
        previous_artifacts = stage_artifacts

    print("Created demo metrics, media, tables, and Artifact lineage.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that you have logged some runs, lets create a report.

    ## 3. Build a Report

    ### Text and image blocks

    Define content with `TableOfContents`, headings, paragraphs, and images. These
    are Report objects, separate from marimo's display elements. The image below is
    from the original Weights & Biases tutorial. Nothing is uploaded while building these lists.
    """)
    return


@app.cell
def _():
    text_blocks = [
        wr.TableOfContents(),
        wr.H1("Text and images example"),
        wr.P("Use a Report to explain an experiment, show results, and share conclusions with your team."),
        wr.Image("https://api.wandb.ai/files/telidavies/images/projects/831572/8ad61fd1.png", caption="Craiyon generated images"),
        wr.P("Panel grids connect this narrative to the underlying experiment data."),
    ]
    return (text_blocks,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Panel grids and run colors

    A `PanelGrid` combines runsets and panels. The first runset selects all runs in
    the project; the second searches for `elephant`, matching the demo media run.
    Panels show validation accuracy, accuracy bars, images, and differing run settings.

    `custom_run_colors` maps **run IDs** to colors. The optional demo step records
    those IDs as it creates runs. For existing data, edit the mapping with IDs from
    your own project. The `WeaveBlockArtifact` below embeds Artifact lineage from
    the second project; its API name refers to a Report block, not LLM tracing.
    """)
    return


@app.cell
def _(demo_run_colors, entity, project):
    panel_grid = wr.PanelGrid(
        runsets=[
            wr.Runset(entity, project, "All runs"),
            wr.Runset(entity, project, "Elephants only", query="elephant"),
        ],
        panels=[
            wr.LinePlot(x="Step", y=["val_acc"], smoothing_factor=0.8),
            wr.BarPlot(metrics=["acc"]),
            wr.MediaBrowser(media_keys=["img"], num_columns=1),
            wr.RunComparer(diff_only="split", layout=wr.Layout(w=24, h=9)),
        ],
    )
    panel_grid.custom_run_colors = dict(demo_run_colors)
    artifact_lineage = wr.WeaveBlockArtifact(
        entity=entity, project=project, artifact="model-1", tab="lineage",
    )
    return artifact_lineage, panel_grid


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Assemble, save, load, and update

    `Report(...)` creates a local draft. Assign its block list to define the order of
    content. The publishing cell calls `save()`, reloads the saved object with
    `Report.from_url()`, changes its description, then saves the update to the same
    Report. Later clicks update that saved draft unless you rebuild the draft cell.

    Inspect and edit the content first, then click **Publish the Quickstart Report**.
    This is the first step that publishes a Report to Weights & Biases.
    """)
    return


@app.cell
def _(artifact_lineage, entity, panel_grid, project, text_blocks):
    quickstart_report = wr.Report(
        entity=entity, project=project, title="Quickstart Report",
        description="A Report built with the Weights & Biases Report API",
    )
    quickstart_report.blocks = [
        text_blocks[0], wr.H1("Panel grid example"), panel_grid,
        wr.H1("Artifact lineage example"), artifact_lineage, *text_blocks[1:],
    ]

    quickstart_report.save()
    loaded_report = wr.Report.from_url(quickstart_report.url)
    loaded_report.description = "Created, loaded, and updated with the Weights & Biases Report API"
    loaded_report.save()
    return (quickstart_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. FAQ examples: build more local drafts

    ### Resize panels

    Report width can be `readable`, `fixed`, or `fluid`. The publishing example below
    will clone the main Report and set its width to `fluid`.

    Panel layout controls position (`x`, `y`) and dimensions (`w`, `h`) in grid units.
    The following parallel coordinates plot uses the full 24-unit panel-grid width.
    Configuration fields use a `c::` prefix; the demo runs log `encoder` and `optimizer`.
    """)
    return


@app.cell
def _(entity, project):
    resize_report = wr.Report(
            entity=entity,
            project=project,
            title="Resizing panels",
            description="Look at this wide parallel coordinates plot!",
            blocks=[
                wr.PanelGrid(
                    panels=[
                        wr.ParallelCoordinatesPlot(
                            columns=[
                                wr.ParallelCoordinatesPlotColumn(metric="Step"),
                                wr.ParallelCoordinatesPlotColumn(metric="c::encoder"),
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
    return (resize_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Block gallery

    `wr.blocks` exposes the available block classes. This draft demonstrates headings,
    formatted text, inline code and LaTeX, code blocks, tables, Artifact views, images,
    lists, quotes, callouts, separators, and video. The Artifact table view references
    `model-1:v0` to select an exact version.
    """)
    return


@app.cell
def _(entity, project):
    block_gallery_report = wr.Report(
            entity=entity,
            project=project,
            title="Weights & Biases Block Gallery",
            description="Check out all of the blocks available in Weights & Biases",
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
                wr.WeaveBlockSummaryTable(entity=entity, project=project, table_name="my-table"),
                wr.WeaveBlockArtifact(entity=entity, project=project, artifact="model-1", tab="lineage"),
                wr.WeaveBlockArtifactVersionedFile(entity=entity, project=project, artifact="model-1", version="v0", file="dataframe.table.json"),
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
    return (block_gallery_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Panel gallery

    `wr.panels` exposes chart and comparison panel classes. This gallery includes
    line, scatter, bar, scalar, parallel coordinates, parameter importance, media,
    code/run comparison, and custom charts. It also demonstrates grouping, smoothing,
    axis settings, and legends. Adjust these settings for the metrics in your project.

    Embedded summary tables and Artifact views belong directly in `report.blocks`,
    while charts belong inside `PanelGrid.panels`.
    """)
    return


@app.cell
def _(entity, project):
    panel_gallery_report = wr.Report(
            entity=entity,
            project=project,
            title="Weights & Biases Panel Gallery",
            description="Check out all of the panels available in Weights & Biases",
            width="fluid",
            blocks=[
                wr.PanelGrid(
                    runsets=[wr.Runset(entity, project, "Metrics and media"), wr.Runset(entity, project, "Artifacts")],
                    panels=[
                        wr.MediaBrowser(media_keys=["img"]),
                        wr.MarkdownPanel(markdown="Hello *italic* **bold** $e=mc^2$ `something`"),

                        # LinePlot with various settings enabled
                        wr.LinePlot(
                            title="Validation Accuracy over Time",
                            x="Step",
                            y=["val_acc"],
                            range_x=(1, 100),
                            range_y=(0, 1),
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
                        wr.BarPlot(title="Validation Loss by Encoder", metrics=["val_loss"], orientation="h", range_x=(0, 2), title_x="Validation Loss", groupby="encoder", groupby_aggfunc="median", groupby_rangefunc="stddev", max_runs_to_show=20, max_bars_to_show=3, font_size="auto"),
                        wr.ScalarChart(title="Maximum Number of Steps", metric="Step", groupby_aggfunc="max", groupby_rangefunc="stderr", font_size="large"),
                        wr.CodeComparer(diff="split"),
                        wr.ParallelCoordinatesPlot(columns=[wr.ParallelCoordinatesPlotColumn("Step"), wr.ParallelCoordinatesPlotColumn("c::encoder"), wr.ParallelCoordinatesPlotColumn("c::optimizer"), wr.ParallelCoordinatesPlotColumn("val_acc"), wr.ParallelCoordinatesPlotColumn("val_loss")]),
                        wr.ParameterImportancePlot(with_respect_to="val_loss"),
                        wr.RunComparer(diff_only=True),
                        wr.CustomChart(query={"summary": ["val_loss", "val_acc"]}, chart_name="wandb/scatter/v0", chart_fields={"x": "val_loss", "y": "val_acc"}),
                    ],
                ),
                # Add WeaveBlock types directly to the blocks list
                wr.WeaveBlockSummaryTable(entity=entity, project=project, table_name="my-table"),
                wr.WeaveBlockArtifact(entity=entity, project=project, artifact="model-1", tab="lineage"),
                wr.WeaveBlockArtifactVersionedFile(entity=entity, project=project, artifact="model-1", version="v0", file="dataframe.table.json"),
            ],
        )
    return (panel_gallery_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Related Reports, grouped colors, and combined blocks

    These two drafts use the original tutorial's public CleanRL runs in
    `openrlbenchmark/cleanrl`. They show reinforcement-learning metrics and videos;
    viewing them requires access to that project. `RunsetGroup` assigns colors to
    specific configuration groups, using each configuration field as the key and
    its observed setting as the value.

    You can combine two Reports by concatenating their block lists. Publishing later
    also creates a Report with `wr.Link` objects pointing to these two saved Reports;
    `MarkdownBlock` supports ordinary Markdown links as well.
    """)
    return


@app.cell
def _(entity, project):
    related_report_one = wr.Report(
            entity=entity,
            project=project,
            title="Report 1",
            description="Great content coming from Report 1",
            blocks=[
                wr.H1(text="Heading from Report 1"),
                wr.P(text="Compare the public CleanRL experiments using the panels below."),
                wr.PanelGrid(
                    panels=[
                        wr.LinePlot(title="Episodic Return", x="global_step", y=["charts/episodic_return"], smoothing_factor=0.85, groupby_aggfunc="mean", groupby_rangefunc="minmax", layout=wr.Layout(x=0, y=0, w=12, h=8)),
                        wr.MediaBrowser(media_keys=["videos"], num_columns=4, layout=wr.Layout(w=12, h=8)),
                    ],
                    runsets=[wr.Runset(entity="openrlbenchmark", project="cleanrl", query="bigfish", groupby=["exp_name"])],
                    custom_run_colors={
                        wr.RunsetGroup(runset_name="Run set", keys=(wr.RunsetGroupKey(key=wr.Config("exp_name"), value="ppg_procgen"),)): "#2980b9",
                        wr.RunsetGroup(runset_name="Run set", keys=(wr.RunsetGroupKey(key=wr.Config("exp_name"), value="ppo_procgen"),)): "#e74c3c",
                    },
                ),
            ],
        )

    related_report_two = wr.Report(
            entity=entity,
            project=project,
            title="Report 2",
            description="Great content coming from Report 2",
            blocks=[
                wr.H1(text="Heading from Report 2"),
                wr.P(text="Compare the public CleanRL experiments using the panels below."),
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
                        wr.RunsetGroup(runset_name="DQN", keys=(wr.RunsetGroupKey(key=wr.Config("exp_name"), value="dqn_atari"),)): "#e84118",
                        wr.RunsetGroup(runset_name="SAC-discrete 0.8", keys=(wr.RunsetGroupKey(key=wr.Config("exp_name"), value="sac_atari"),)): "#fbc531",
                        wr.RunsetGroup(runset_name="SAC-discrete 0.88", keys=(wr.RunsetGroupKey(key=wr.Config("exp_name"), value="sac_atari"),)): "#00a8ff",
                    },
                ),
            ],
        )
    return related_report_one, related_report_two


@app.cell
def _(entity, project, related_report_one, related_report_two):
    combined_report = wr.Report(
        entity=entity, project=project, title="Combined blocks report",
        description="Blocks from both related Reports",
        blocks=[*related_report_one.blocks, *related_report_two.blocks],
    )
    return (combined_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Publish the FAQ examples

    The next action loads the saved Quickstart Report into a fresh object, changes
    its width, and calls `save(clone=True)` to create a separate copy. Loading a fresh
    object matters: `save(clone=True)` changes the saved object's ID in the current
    library, so cloning the original draft directly would change its save target.

    It then saves the six drafts above and creates a links Report after both related
    Report URLs exist. On the first click this creates eight FAQ Reports. Later
    clicks update the six saved drafts and create another width clone and links
    Report. Save the Quickstart Report first.
    """)
    return


@app.cell
def _(
    block_gallery_report,
    combined_report,
    entity,
    panel_gallery_report,
    project,
    quickstart_report,
    related_report_one,
    related_report_two,
    resize_report,
):
    wide_report = wr.Report.from_url(quickstart_report.url)
    wide_report.width = "fluid"
    wide_report.save(clone=True)

    for faq_draft in [
        resize_report, block_gallery_report, panel_gallery_report,
        related_report_one, related_report_two, combined_report,
    ]:
        faq_draft.save()

    links_report = wr.Report(
        entity=entity, project=project, title="Report with links",
        description="Link related Reports with rich text or Markdown",
        blocks=[
            wr.H1(["Results from ", wr.Link("Report 1", url=related_report_one.url)]),
            wr.P(["Compare these results with ", wr.Link("Report 2", url=related_report_two.url)]),
            wr.MarkdownBlock(f"Also visit [Report 1]({related_report_one.url}) and [Report 2]({related_report_two.url})."),
        ],
    )
    links_report.save()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Update lists of blocks explicitly

    A Report contains lists such as `report.blocks`, `panel_grid.panels`, and
    `panel_grid.runsets`. Object-reference behavior can vary across library versions.
    To make an update explicit, edit the block and then reassign the block list.
    Keep related mutations in one cell so marimo doesn't need to infer their order.

    The following example stays local: it never calls `save()`.
    """)
    return


@app.cell
def _(entity, project):
    assignment_report = wr.Report(entity=entity, project=project)
    heading = wr.H1(text=["Hello", " World!"])
    assignment_report.blocks = [heading]
    heading.text = ["Something", " New"]
    assignment_report.blocks = [heading]
    print(assignment_report.blocks[0].text)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify the saved Reports

    Open the URLs printed by the publishing cells and check the narrative, panels,
    Artifact lineage, and FAQ examples. If a panel is empty, confirm that its runset
    points to a project with the requested metrics or media keys. If you skipped the
    demo data, adapt the runsets and Artifact names to your existing data.

    Inspect `wr.blocks` and `wr.panels` in your editor for more options. Edit local
    drafts first, then publish only the examples you want to keep.
    """)
    return


if __name__ == "__main__":
    app.run()
