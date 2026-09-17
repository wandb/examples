# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "colabfold[alphafold]>=1.6.3,<1.7",
#     "jax[cuda12]>=0.6.2,<0.12; sys_platform == 'linux'",
#     "marimo>=0.24.0",
#     "matplotlib>=3.8,<4",
#     "numpy>=2.0,<3",
#     "py3Dmol>=2.0,<3",
#     "wandb>=0.19.10",
# ]
# ///

"""Predict a protein structure with ColabFold and log it to W&B.

Run:

    uvx marimo edit alphafold_with_w_b_align_fold_log.py --sandbox
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="ColabFold with W&B")


@app.cell
def _():
    import json
    import os
    import re
    import subprocess
    import sys
    import time
    import uuid
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import py3Dmol
    import wandb

    return (
        Path,
        json,
        mo,
        np,
        os,
        plt,
        py3Dmol,
        re,
        subprocess,
        sys,
        time,
        uuid,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/alphafold-with-w-b-align-fold-log/alphafold_with_w_b_align_fold_log.py/server)

    <img src="https://i.imgur.com/4ta7Arm.png" alt="Weights & Biases" />

    # ColabFold with W&B: Align, Fold, Log

    This notebook predicts a protein structure with
    [ColabFold](https://github.com/sokrypton/ColabFold), then logs the sequence,
    alignment summary, confidence plots, timings, and predicted molecule to
    Weights & Biases.

    It preserves the **align → fold → log** lesson from
    [DeepMind's original AlphaFold Colab](https://github.com/google-deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb)
    and the W&B adaptation described in
    [this Report](https://wandb.me/alphafold-short-report). ColabFold replaces
    the retired Colab setup with an MMseqs2 alignment service and local
    AlphaFold2 inference.

    No Docker image or local genetic database is required. The prediction runs
    on the notebook GPU; only the submitted sequence is sent to the public
    ColabFold MSA service.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Runtime

    Open this notebook in a **GPU-backed molab session**. Its Python
    dependencies include ColabFold, the AlphaFold2 inference package, and a
    CUDA-enabled JAX build.

    The first prediction downloads the model parameters and can take longer
    than later runs. ColabFold queries its shared public MSA service for this
    small demonstration, so submit one sequence at a time from this notebook.
    Opening the notebook does not query the service, download weights, run a
    prediction, or create W&B objects.
    """)
    return


@app.cell(hide_code=True)
def _(mo, subprocess, sys):
    _linux_available = sys.platform.startswith("linux")
    try:
        _gpu_probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        _gpu_names = [
            _name.strip() for _name in _gpu_probe.stdout.splitlines() if _name.strip()
        ]
    except (FileNotFoundError, subprocess.CalledProcessError):
        _gpu_names = []

    colabfold_runtime_ready = bool(_linux_available and _gpu_names)
    if colabfold_runtime_ready:
        _runtime_message = mo.callout(
            mo.md(f"Ready to fold on **{', '.join(_gpu_names)}**."),
            kind="success",
        )
    else:
        _runtime_message = mo.callout(
            mo.md(
                "For a practical prediction, reopen this notebook in a "
                "GPU-backed molab session."
            ),
            kind="warn",
            title="GPU session needed",
        )
    _runtime_message
    return (colabfold_runtime_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to
    use `WANDB_API_KEY` from the molab Secrets panel or credentials already
    stored in this runtime. A fresh molab session does not inherit credentials
    from your local computer.

    The entity is the team name in a W&B project URL:
    `wandb.ai/<entity>/<project>`. Leave it blank to use your default entity.
    Merely editing these fields does not authenticate or create a run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="W&B API key (optional)",
                placeholder="Paste a key or use configured credentials",
                full_width=True,
            ),
            entity=mo.ui.text(
                label="W&B entity or team (optional)",
                placeholder="Leave blank to use your default entity",
                full_width=True,
            ),
            project=mo.ui.text(
                value="alphafold",
                label="W&B project",
                full_width=True,
            ),
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(mo, wandb, wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Submit the authentication form before logging a prediction."),
            kind="info",
        ),
    )

    _submitted_login = wandb_login_form.value
    _api_key = _submitted_login["api_key"].strip()
    _requested_entity = _submitted_login["entity"].strip()
    _project = _submitted_login["project"].strip() or "alphafold"
    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
        )
        _resolved_entity = _requested_entity or wandb.Api().default_entity
        _login_error = None
    except (wandb.errors.Error, ValueError) as _error:
        _login_ok = False
        _resolved_entity = None
        _login_error = str(_error)

    mo.stop(
        not _login_ok or not _resolved_entity,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key and "
                "entity, then submit again.\n\n"
                f"W&B reported: `{_login_error or 'No default entity was found.'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {"entity": _resolved_entity, "project": _project}
    mo.callout(
        mo.md(
            f"Connected to W&B. Results will target "
            f"`{_resolved_entity}/{_project}` only after you submit the log form."
        ),
        kind="success",
    )
    return (wandb_settings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Enter a sequence and make a prediction

    Choose a sample protein or paste a custom amino-acid sequence. ColabFold
    first creates a multiple-sequence alignment with MMseqs2, then predicts the
    structure locally with AlphaFold2. A short sequence can still take several
    minutes because JAX compiles the model on its first run.

    ### Sample sequences

    You can also find protein sequences in
    [RCSB PDB](https://www.rcsb.org/),
    [UniProt](https://www.uniprot.org/), or
    [EMBL-EBI](https://www.ebi.ac.uk/).
    """)
    return


@app.cell
def _():
    SEQUENCES = [
        {
            "name": "lithostathine",
            "species": "human",
            "desc": "small and fast",
            "url": "https://www.ebi.ac.uk/pdbe/entry/pdb/1lit/protein/1",
            "seq": "QEAQTELPQARISCPEGTNAYRSYCYYFNEDRETWVDADLYCQNMNSGNLVSVLTQAEGAFVASLIKESGTDDFNVWIGLHDPKKNRRWHWSSGSLVSYKSWGIGAPSSVNPGYCVSLTSSTGFQKWKDVPCEDKFSFVCKFKN",
        },
        {
            "name": "crystallin",
            "species": "cow",
            "desc": "eye lens protein",
            "url": "https://www.rcsb.org/structure/1AMM",
            "seq": "GKITFYEDRGFQGHCYECSSDCPNLQPYFSRCNSIRVDSGCWMLYERPNYQGHQYFLRRGDYPDYQQWMGFNDSIRSCRLIPQHTGTFRMRIYERDDFRGQMSEITDDCPSLQDRFHLTEVHSLNVLEGSWVLYEMPSYRGRQYLLRPGEYRRYLDWGAMNAKVGSLRRVMDFY",
        },
        {
            "name": "olfactory",
            "species": "mouse",
            "desc": "smell-detecting protein",
            "url": "https://www.rcsb.org/structure/1JOD",
            "seq": "AEDGPQKQQLEMPLVLDQDLTQQMRLRVESLKQRGEKKQDGEKLIRPAESVYRLDFIQQQKLQFDHWNVVLDKPGKVTITGTSQNWTPDLTNLMTRQLLDPAAIFWRKEDSDAMDWNEADALEFGERLSDLAKIRKVMYFLITFGEGVEPANLKASVVFNQL",
        },
        {
            "name": "fly_thioredoxin",
            "species": "drosophila",
            "desc": "short, comparable in humans",
            "url": "https://www.rcsb.org/structure/1XWC",
            "seq": "MVYQVKDKADLDGQLTKASGKLVVLDFFATWCGPCKMISPKLVELSTQFADNVVVLKVDVDECEDIAMEYNISSMPTFVFLKNGVKVEEFAGANAKRLEDVIKANI",
        },
        {
            "name": "neurotrophin",
            "species": "human",
            "desc": "nice symmetry",
            "url": "https://www.ebi.ac.uk/pdbe/entry/pdb/1b8k/protein/1",
            "seq": "YAEHKSHRGEYSVCDSESLWVTDKSSAIDIRGHQVTVLGEIKTGNSPVKQYFYETRCKEARPVKNGCRGIDDKHWNSQCKTSQTYVRALTSENNKLVGWRWIRIDTSCVCALSRKIGRT",
        },
    ]
    return (SEQUENCES,)


@app.cell(hide_code=True)
def _(SEQUENCES, mo):
    _sample_rows = [
        {
            "name": _sample["name"],
            "species": _sample["species"],
            "description": _sample["desc"],
            "residues": len(_sample["seq"]),
            "reference": _sample["url"],
        }
        for _sample in SEQUENCES
    ]
    mo.ui.table(_sample_rows)
    return


@app.function
def normalize_sequence(raw_sequence):
    min_sequence_length = 16
    max_sequence_length = 2000

    sequence = raw_sequence.translate(str.maketrans("", "", " \n\t")).upper()
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    if not set(sequence).issubset(amino_acids):
        invalid = set(sequence) - amino_acids
        raise ValueError(
            "Input sequence contains non-amino acid letters: "
            f"{invalid}. Use the 20 standard amino-acid codes."
        )
    if len(sequence) < min_sequence_length:
        raise ValueError(
            f"Input sequence is too short: {len(sequence)} residues; "
            f"the minimum is {min_sequence_length}."
        )
    if len(sequence) > max_sequence_length:
        raise ValueError(
            f"Input sequence is too long for this notebook: {len(sequence)} "
            f"residues; the limit is {max_sequence_length}."
        )
    return sequence


@app.function
def safe_sequence_name(name):
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-")
    return normalized or "custom-sequence"


@app.cell(hide_code=True)
def _(SEQUENCES, mo):
    colabfold_form = (
        mo.md(
            "{sample_name}\n\n{custom_sequence}\n\n{msa_mode}\n\n"
            "{num_models}\n\n{num_recycles}\n\n{use_templates}\n\n"
            "{output_dir}\n\n{accept_server}"
        )
        .batch(
            sample_name=mo.ui.dropdown(
                options=[_sample["name"] for _sample in SEQUENCES],
                value="lithostathine",
                label="Sample protein",
                full_width=True,
            ),
            custom_sequence=mo.ui.text_area(
                value="",
                label="Custom sequence (optional; overrides the sample)",
                placeholder="Paste amino-acid letters only",
                full_width=True,
            ),
            msa_mode=mo.ui.dropdown(
                options=["mmseqs2_uniref_env", "mmseqs2_uniref"],
                value="mmseqs2_uniref_env",
                label="MSA database selection",
                full_width=True,
            ),
            num_models=mo.ui.dropdown(
                options=["1", "3", "5"],
                value="1",
                label="Number of AlphaFold2 models",
                full_width=True,
            ),
            num_recycles=mo.ui.dropdown(
                options=["3", "6", "12"],
                value="3",
                label="Prediction recycles",
                full_width=True,
            ),
            use_templates=mo.ui.checkbox(
                value=False,
                label="Search for structural templates",
            ),
            output_dir=mo.ui.text(
                value="/tmp/colabfold-runs",
                label="Output root",
                full_width=True,
            ),
            accept_server=mo.ui.checkbox(
                value=False,
                label=(
                    "I will submit this single query to the shared ColabFold "
                    "MSA service and follow its fair-use guidance"
                ),
            ),
        )
        .form(
            submit_button_label="Run alignment and structure prediction",
            bordered=True,
        )
    )
    colabfold_form
    return (colabfold_form,)


@app.function
def run_colabfold(request):
    base_output_dir = Path(request["output_dir"]).expanduser().resolve()
    run_id = uuid.uuid4().hex[:10]
    run_root = base_output_dir / f"{request['sequence_id']}-{run_id}"
    prediction_dir = run_root / "prediction"
    prediction_dir.mkdir(parents=True, exist_ok=False)

    fasta_path = run_root / f"{request['sequence_id']}.fasta"
    fasta_path.write_text(
        f">{request['sequence_id']}\n{request['sequence']}\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "colabfold.batch",
        str(fasta_path),
        str(prediction_dir),
        "--model-type",
        "alphafold2_ptm",
        "--msa-mode",
        request["msa_mode"],
        "--num-models",
        str(request["num_models"]),
        "--num-recycle",
        str(request["num_recycles"]),
        "--num-seeds",
        "1",
        "--num-relax",
        "0",
        "--rank",
        "plddt",
        "--compile-mode",
        "fast",
        "--overwrite-existing-results",
    ]
    if request["use_templates"]:
        command.append("--templates")

    command_environment = os.environ.copy()
    compilation_cache = Path("/tmp/colabfold-jax-cache")
    compilation_cache.mkdir(parents=True, exist_ok=True)
    command_environment.setdefault("JAX_COMPILATION_CACHE_DIR", str(compilation_cache))

    started_at = time.monotonic()
    subprocess.run(command, env=command_environment, check=True)
    wall_time_seconds = time.monotonic() - started_at

    relaxed_candidates = sorted(
        prediction_dir.glob(f"{request['sequence_id']}_relaxed_rank_001_*.pdb")
    )
    unrelaxed_candidates = sorted(
        prediction_dir.glob(f"{request['sequence_id']}_unrelaxed_rank_001_*.pdb")
    )
    structure_candidates = relaxed_candidates or unrelaxed_candidates
    if not structure_candidates:
        raise FileNotFoundError(
            f"ColabFold finished without a top-ranked PDB file in {prediction_dir}."
        )

    score_candidates = sorted(
        prediction_dir.glob(f"{request['sequence_id']}_scores_rank_001_*.json")
    )
    if not score_candidates:
        raise FileNotFoundError(
            "ColabFold finished without a top-ranked score JSON file in "
            f"{prediction_dir}."
        )

    alignment_path = prediction_dir / f"{request['sequence_id']}.a3m"
    if not alignment_path.is_file():
        raise FileNotFoundError(
            f"ColabFold finished without the expected MSA file: {alignment_path}."
        )

    return {
        **request,
        "alignment_path": str(alignment_path),
        "fasta_path": str(fasta_path),
        "prediction_dir": str(prediction_dir),
        "pred_output_path": str(structure_candidates[0]),
        "score_path": str(score_candidates[0]),
        "wall_time_seconds": wall_time_seconds,
    }


@app.cell
def _(SEQUENCES, colabfold_form, colabfold_runtime_ready, mo):
    mo.stop(
        colabfold_form.value is None,
        mo.callout(
            mo.md(
                "Review the inputs, then submit the form when you are ready "
                "to query the MSA service and start the GPU prediction."
            ),
            kind="info",
        ),
    )
    mo.stop(
        not colabfold_form.value["accept_server"],
        mo.callout(
            mo.md(
                "Confirm the shared MSA service guidance in the form before "
                "submitting the sequence."
            ),
            kind="warn",
        ),
    )
    mo.stop(
        not colabfold_runtime_ready,
        mo.callout(
            mo.md("Attach a GPU-backed molab runtime, then submit again."),
            kind="warn",
        ),
    )

    _submitted_fold = colabfold_form.value
    _selected_sample = next(
        _sample
        for _sample in SEQUENCES
        if _sample["name"] == _submitted_fold["sample_name"]
    )
    _custom_sequence = _submitted_fold["custom_sequence"].strip()
    _sequence = normalize_sequence(_custom_sequence or _selected_sample["seq"])
    _sequence_name = "custom-sequence" if _custom_sequence else _selected_sample["name"]
    _metadata = (
        {
            "name": "custom-sequence",
            "species": "",
            "desc": "user-provided amino-acid sequence",
            "url": "",
        }
        if _custom_sequence
        else {key: _selected_sample[key] for key in ("name", "species", "desc", "url")}
    )
    _fold_request = {
        "metadata": _metadata,
        "model_type": "alphafold2_ptm",
        "msa_mode": _submitted_fold["msa_mode"],
        "num_models": int(_submitted_fold["num_models"]),
        "num_recycles": int(_submitted_fold["num_recycles"]),
        "output_dir": _submitted_fold["output_dir"].strip() or "/tmp/colabfold-runs",
        "sequence": _sequence,
        "sequence_id": safe_sequence_name(_sequence_name),
        "use_templates": _submitted_fold["use_templates"],
    }
    fold_result = run_colabfold(_fold_request)
    return (fold_result,)


@app.cell(hide_code=True)
def _(fold_result, mo):
    mo.callout(
        mo.md(
            "ColabFold finished. The complete output is at "
            f"`{fold_result['prediction_dir']}`."
        ),
        kind="success",
    )
    return


@app.function
def read_a3m_sequences(alignment_path):
    sequences = []
    current = []
    for raw_line in alignment_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().replace("\x00", "")
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            continue
        current.append("".join(letter for letter in line if not letter.islower()))
    if current:
        sequences.append("".join(current))
    return sequences


@app.function
def inspect_prediction(fold_result):
    prediction_dir = Path(fold_result["prediction_dir"])
    pred_output_path = Path(fold_result["pred_output_path"])
    score_path = Path(fold_result["score_path"])
    alignment_path = Path(fold_result["alignment_path"])

    scores = json.loads(score_path.read_text(encoding="utf-8"))
    plddt = [float(value) for value in scores.get("plddt", [])]
    if not plddt:
        for line in pred_output_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                plddt.append(float(line[60:66]))
    if not plddt:
        raise ValueError(f"No per-residue pLDDT values were found in {score_path}.")

    alignment_sequences = read_a3m_sequences(alignment_path)
    if not alignment_sequences:
        raise ValueError(f"No sequences were found in {alignment_path}.")
    alignment_width = len(fold_result["sequence"])
    normalized_alignment = [
        sequence[:alignment_width].ljust(alignment_width, "-")
        for sequence in alignment_sequences
    ]
    coverage = np.array(
        [
            sum(sequence[index] not in {"-", "."} for sequence in normalized_alignment)
            for index in range(alignment_width)
        ]
    )

    msa_plot_path = prediction_dir / "msa_coverage.png"
    msa_figure, msa_axis = plt.subplots(figsize=(12, 3))
    msa_axis.set_title("Per-Residue Count of Non-Gap Amino Acids in the MSA")
    msa_axis.plot(coverage, color="black")
    msa_axis.set_xlabel("Residue")
    msa_axis.set_ylabel("Non-gap count")
    msa_figure.tight_layout()
    msa_figure.savefig(msa_plot_path, dpi=150)
    plt.close(msa_figure)

    confidence_plot_path = prediction_dir / "plddt_confidence.png"
    confidence_figure, confidence_axis = plt.subplots(figsize=(10, 4))
    confidence_axis.plot(plddt, color="#0053D6")
    confidence_axis.set_ylim(0, 100)
    confidence_axis.set_xlabel("Residue")
    confidence_axis.set_ylabel("pLDDT")
    confidence_axis.set_title("Predicted Local Distance Difference Test")
    confidence_figure.tight_layout()
    confidence_figure.savefig(confidence_plot_path, dpi=150)
    plt.close(confidence_figure)

    predicted_aligned_error = scores.get("pae")
    pae_plot_path = None
    if predicted_aligned_error:
        pae_array = np.asarray(predicted_aligned_error, dtype=float)
        max_pae = float(scores.get("max_pae", np.max(pae_array)))
        pae_plot_path = prediction_dir / "predicted_aligned_error.png"
        pae_figure, pae_axis = plt.subplots(figsize=(6, 5))
        pae_image = pae_axis.imshow(
            pae_array,
            vmin=0,
            vmax=max_pae,
            cmap="Greens_r",
        )
        pae_figure.colorbar(pae_image, ax=pae_axis)
        pae_axis.set_title("Predicted Aligned Error")
        pae_axis.set_xlabel("Scored residue")
        pae_axis.set_ylabel("Aligned residue")
        pae_figure.tight_layout()
        pae_figure.savefig(pae_plot_path, dpi=150)
        plt.close(pae_figure)

    unique_alignment_sequences = len(set(alignment_sequences))
    return {
        **fold_result,
        "confidence_plot_path": str(confidence_plot_path),
        "mean_plddt": float(np.mean(plddt)),
        "msa_plot_path": str(msa_plot_path),
        "msa_sequences": unique_alignment_sequences,
        "pae_plot_path": str(pae_plot_path) if pae_plot_path else None,
        "plddt": plddt,
        "ptm": float(scores["ptm"]) if "ptm" in scores else None,
    }


@app.cell
def _(fold_result):
    prediction_summary = inspect_prediction(fold_result)
    return (prediction_summary,)


@app.cell
def _(Path, prediction_summary, py3Dmol):
    _pdb_text = Path(prediction_summary["pred_output_path"]).read_text(encoding="utf-8")
    structure_view = py3Dmol.view(width=800, height=600)
    structure_view.addModel(_pdb_text, "pdb")
    structure_view.setStyle(
        {
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "roygb",
                    "min": 0,
                    "max": 100,
                }
            }
        }
    )
    structure_view.zoomTo()
    structure_view
    return


@app.cell(hide_code=True)
def _(mo, prediction_summary):
    _result_plots = [
        mo.image(prediction_summary["msa_plot_path"]),
        mo.image(prediction_summary["confidence_plot_path"]),
    ]
    if prediction_summary["pae_plot_path"]:
        _result_plots.append(mo.image(prediction_summary["pae_plot_path"]))
    mo.vstack(
        [
            mo.md(
                "**pLDDT confidence bands:** very low (0–50), low (50–70), "
                "confident (70–90), very high (90–100)."
            ),
            mo.hstack(_result_plots, widths="equal", wrap=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(Path, mo, prediction_summary):
    _pdb_path = Path(prediction_summary["pred_output_path"])
    _ptm_text = (
        f"  \npTM: **{prediction_summary['ptm']:.2f}**"
        if prediction_summary["ptm"] is not None
        else ""
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"Mean top-ranked pLDDT: **{prediction_summary['mean_plddt']:.1f}**"
                    f"{_ptm_text}  \n"
                    f"MSA sequences: **{prediction_summary['msa_sequences']:,}**  \n"
                    f"Output directory: `{prediction_summary['prediction_dir']}`"
                ),
                kind="info",
            ),
            mo.download(
                data=_pdb_path.read_bytes(),
                filename=_pdb_path.name,
                label="Download the top-ranked PDB",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the prediction

    pLDDT describes local confidence; a high value does not by itself establish
    confidence in relative domain placement or biological function. Predicted
    aligned error describes confidence in the relative positions of residue
    pairs. See the
    [AlphaFold methods paper](https://www.nature.com/articles/s41586-021-03819-2)
    and the [AlphaFold FAQ](https://alphafold.ebi.ac.uk/faq) for guidance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Log to W&B: Upload the sequence, 3D structure, and results

    The original tutorial logs sequence metadata, alignment statistics,
    pipeline timing, the MSA plot, confidence plots, and the predicted molecule.
    The Table below keeps those fields together so runs are easy to compare.

    Submitting the form creates one W&B run and performs one `run.log` call.
    Opening the notebook or changing an unsubmitted run name does nothing
    remotely.
    """)
    return


@app.function
def time_units(duration):
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"


@app.function
def log_prediction(settings, prediction_summary, run_name):
    metadata = prediction_summary["metadata"]
    run_config = {
        "sequence": prediction_summary["sequence"],
        "sequence_length": len(prediction_summary["sequence"]),
        "species": metadata["species"],
        "description": metadata["desc"],
        "reference_url": metadata["url"],
        "model_type": prediction_summary["model_type"],
        "msa_mode": prediction_summary["msa_mode"],
        "num_models": prediction_summary["num_models"],
        "num_recycles": prediction_summary["num_recycles"],
        "use_templates": prediction_summary["use_templates"],
    }
    numeric_metrics = {
        "mean_plddt": prediction_summary["mean_plddt"],
        "msa_sequences": prediction_summary["msa_sequences"],
        "wall_time_seconds": prediction_summary["wall_time_seconds"],
    }
    if prediction_summary["ptm"] is not None:
        numeric_metrics["ptm"] = prediction_summary["ptm"]

    with wandb.init(
        entity=settings["entity"],
        project=settings["project"],
        name=run_name or None,
        config=run_config,
        job_type="protein-structure-prediction",
        reinit="create_new",
    ) as run:
        molecule = wandb.Molecule(prediction_summary["pred_output_path"])
        msa_plot = wandb.Image(prediction_summary["msa_plot_path"])
        confidence_plot = wandb.Image(prediction_summary["confidence_plot_path"])
        pae_plot = (
            wandb.Image(prediction_summary["pae_plot_path"])
            if prediction_summary["pae_plot_path"]
            else None
        )
        prediction_table = wandb.Table(
            columns=[
                "tag",
                "species",
                "sequence",
                "molecule",
                "MSA coverage",
                "confidence",
                "predicted aligned error",
            ]
        )
        prediction_table.add_data(
            metadata["name"],
            metadata["species"],
            prediction_summary["sequence"],
            molecule,
            msa_plot,
            confidence_plot,
            pae_plot,
        )
        log_payload = {
            **numeric_metrics,
            "wall_time": time_units(prediction_summary["wall_time_seconds"]),
            "view_3D": molecule,
            "msa_coverage": msa_plot,
            "plddt_confidence": confidence_plot,
            "predicted_molecules": prediction_table,
        }
        if pae_plot is not None:
            log_payload["predicted_aligned_error"] = pae_plot
        run.log(log_payload)
        run_url = run.url

    return {
        "run_url": run_url,
        "run_path": f"{settings['entity']}/{settings['project']}",
    }


@app.cell(hide_code=True)
def _(mo, prediction_summary, wandb_settings):
    _log_context = mo.md(
        f"Prediction: `{prediction_summary['metadata']['name']}`  \n"
        f"W&B target: `{wandb_settings['entity']}/{wandb_settings['project']}`"
    )
    prediction_log_form = (
        mo.md("{run_name}")
        .batch(
            run_name=mo.ui.text(
                value=f"colabfold-{prediction_summary['sequence_id']}",
                label="W&B run name (optional)",
                full_width=True,
            )
        )
        .form(submit_button_label="Log prediction to W&B", bordered=True)
    )
    mo.vstack([_log_context, prediction_log_form])
    return (prediction_log_form,)


@app.cell(hide_code=True)
def _(mo, prediction_log_form):
    mo.stop(
        prediction_log_form.value is None,
        mo.callout(
            mo.md(
                "Submit the form above when you are ready to create one W&B "
                "run and upload the prediction."
            ),
            kind="info",
        ),
    )
    prediction_log_request = {"run_name": prediction_log_form.value["run_name"].strip()}
    return (prediction_log_request,)


@app.cell
def _(prediction_log_request, prediction_summary, wandb_settings):
    logged_prediction = log_prediction(
        wandb_settings,
        prediction_summary,
        prediction_log_request["run_name"],
    )
    return (logged_prediction,)


@app.cell(hide_code=True)
def _(logged_prediction, mo):
    mo.callout(
        mo.md(
            "Prediction logged. Open the "
            f"[W&B run]({logged_prediction['run_url']}) and inspect the "
            "workspace charts, run configuration, 3D molecule, and "
            "`predicted_molecules` Table."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify and next steps

    In the W&B run, verify that:

    - `mean_plddt`, `ptm`, `msa_sequences`, and `wall_time_seconds` appear in
      the run history and summary;
    - the 3D molecule, MSA coverage, pLDDT, and predicted-aligned-error panels
      render correctly; and
    - `predicted_molecules` contains the sequence, structure, and plots in one
      comparable row.

    Try another short sample, or increase the number of models and compare its
    confidence and runtime with the first prediction.

    ## FAQ and troubleshooting

    * **Do I need Docker or local databases?** No. ColabFold queries the public
      MMseqs2 MSA service and runs AlphaFold2 inference inside the notebook
      environment.
    * **Why is the first run slower?** It downloads AlphaFold2 model parameters
      and compiles the model for the current GPU and input shape.
    * **Can I submit a batch?** This tutorial intentionally submits one serial
      query. The public MSA server is a shared, rate-limited resource. For large
      workloads, follow ColabFold's local-database instructions instead.
    * **Why did prediction fail despite detecting a GPU?** Restart the molab
      session once after dependency installation, then try the short
      `lithostathine` example. The detailed ColabFold log is saved as
      `log.txt` in the displayed output directory.
    * **Where should I report ColabFold problems?** Use the
      [ColabFold issue tracker](https://github.com/sokrypton/ColabFold/issues).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## License and disclaimer

    ColabFold is community-supported software and is not an officially
    supported Google product. Its code is MIT licensed, while the AlphaFold
    implementation, model parameters, dependencies, and remote services have
    their own licenses and terms. Review the current
    [ColabFold documentation](https://github.com/sokrypton/ColabFold) and
    [AlphaFold license information](https://github.com/google-deepmind/alphafold#license-and-disclaimer)
    before publishing or distributing results.

    Protein-structure predictions are for theoretical modeling, not clinical
    use. Predictions have varying confidence and require scientific review.

    Publications based on these results should cite both the
    [ColabFold paper](https://doi.org/10.1038/s41592-022-01488-1) and the
    [AlphaFold paper](https://doi.org/10.1038/s41586-021-03819-2).
    """)
    return


if __name__ == "__main__":
    app.run()
