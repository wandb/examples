# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "colabfold[alphafold]>=1.6.3,<1.7",
#     "jax[cuda13]>=0.6.2,<0.12; sys_platform == 'linux'",
#     "marimo>=0.24.0",
#     "matplotlib>=3.8,<4",
#     "numpy>=2.0,<3",
#     "py3Dmol>=2.0,<3",
#     "wandb>=0.19.10",
# ]
# ///

"""Predict a protein structure with ColabFold and log it to Weights & Biases."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="ColabFold with Weights & Biases")

with app.setup:
    import json
    import os
    import re
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import py3Dmol
    import wandb


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    # ColabFold with Weights & Biases: Align, Fold, Log

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/alphafold-with-w-b-align-fold-log/alphafold_with_w_b_align_fold_log.py/server)

    <img src="https://i.imgur.com/4ta7Arm.png" alt="Weights & Biases" />

    This lesson uses [ColabFold](https://github.com/sokrypton/ColabFold) to
    align a protein sequence and predict its structure with AlphaFold2. We then
    log the experiment configuration, confidence metrics, plots, and a 3D
    molecule to Weights & Biases. A prediction table keeps the sequence and results together
    for comparison across runs.

    > This notebook was modified from the original produced by DeepMind to add additional Weights & Biases logging features.

    See the [AlphaFold-ed Proteins in Weights & Biases Tables report](https://forge.coreweave.com/wandb/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc) and view an example protein below.

    ## 1. Prepare the Runtime

    Use a Linux-backed molab runtime, or open the notebook with its declared
    dependencies on Linux:

    ```bash
    uvx marimo edit --sandbox alphafold_with_w_b_align_fold_log.py
    ```

    The next cell checks the runtime and chooses GPU or CPU inference. It
    retains the CPU fallback for compute capability 12.0 devices used by this
    example. You can inspect the selected backend before predicting.

    The prediction cell queries the public ColabFold MSA service and downloads
    model parameters on first use. Submit one sequence at a time and follow
    [ColabFold's shared-service guidance](https://github.com/sokrypton/ColabFold).
    No local sequence database or Docker image is required for this workflow.
    """)
    return


@app.cell
def _():
    runtime_ready = sys.platform.startswith("linux")
    try:
        gpu_probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_rows = [row.rsplit(",", maxsplit=1) for row in gpu_probe.stdout.splitlines() if row.strip()]
        gpu_names = [name.strip() for name, capability in gpu_rows]
        compute_capabilities = [capability.strip() for name, capability in gpu_rows]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        gpu_names = []
        compute_capabilities = []

    inference_backend = "cpu" if not gpu_names or "12.0" in compute_capabilities else "gpu"
    print(f"Platform: {sys.platform}; inference backend: {inference_backend}")
    if not runtime_ready:
        print("Structure prediction requires a Linux-backed runtime for this notebook.")
    return inference_backend, runtime_ready


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 2. Authenticate with Weights & Biases

    The next cell calls `wandb.login()` and sets the destination for results.
    Set `entity` to your team name, or leave it as `None` for your default
    entity.

    Configure `WANDB_API_KEY` in your environment or molab Secrets. Get your
    key from [Weights & Biases](https://wandb.ai/authorize). In a terminal-backed session,
    `wandb.login()` can also prompt for a key. A fresh molab session does not
    inherit your local credentials.
    """)
    return


@app.cell
def _():
    entity = "your-entity"
    project = "alphafold"
    authenticated = wandb.login()
    if not authenticated:
        raise RuntimeError("Configure Weights & Biases credentials before running the lesson.")
    return authenticated, entity, project


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 3. Choose a Protein and Prediction Settings

    The next cell lists five sample sequences with reference links. The two
    helper functions that follow validate an amino-acid sequence and make a
    file-safe identifier. We accept the 20 standard amino-acid codes and limit
    sequence length to 16–2,000 residues.

    In the settings cell, choose a sample from `SEQUENCES`, or replace
    `raw_sequence` with your own sequence. A custom sequence receives its own
    metadata rather than inheriting the sample's species and reference.

    For this short demo, we use one model, three recycles, no structural
    templates, and an MSA cap of `64:128`. These settings bound computation;
    they are not a substitute for a thorough structure-prediction study.
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


@app.cell
def _(SEQUENCES):
    sample = SEQUENCES[0]
    raw_sequence = sample["seq"]
    sequence = normalize_sequence(raw_sequence)
    metadata = (
        {key: sample[key] for key in ("name", "species", "desc", "url")}
        if sequence == normalize_sequence(sample["seq"])
        else {"name": "custom-sequence", "species": "", "desc": "user-provided sequence", "url": ""}
    )
    sequence_id = safe_sequence_name(metadata["name"])
    model_type = "alphafold2_ptm"
    msa_mode = "mmseqs2_uniref_env"
    num_models = 1
    num_recycles = 3
    use_templates = False
    output_root = Path("/tmp/colabfold-runs")
    print(f"Protein: {sequence_id}; residues: {len(sequence)}")
    return (
        metadata,
        model_type,
        msa_mode,
        num_models,
        num_recycles,
        output_root,
        sequence,
        sequence_id,
        use_templates,
    )


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 4. Align and Fold

    The following cells provide a start button and run ColabFold's batch command
    using this notebook's Python environment. The command reads a FASTA file, obtains an
    alignment through MMseqs2, and runs AlphaFold2 inference locally. Each
    execution writes to a fresh output directory.

    Click **Align and fold** to start this computation and submit the sequence
    to the shared MSA service. The command's logs show alignment and inference
    progress. When it finishes, we collect the top-ranked PDB structure, score
    JSON, alignment file, and total elapsed time for the Weights & Biases logging step.
    """)
    return


@app.cell
def _(
    authenticated,
    inference_backend,
    metadata,
    model_type,
    msa_mode,
    num_models,
    num_recycles,
    output_root,
    runtime_ready,
    sequence,
    sequence_id,
    use_templates,
):
    if not runtime_ready:
        raise RuntimeError("Run structure prediction in a Linux-backed runtime.")
    if not authenticated:
        raise RuntimeError("Log in to Weights & Biases first.")

    output_root.mkdir(parents=True, exist_ok=True)
    prediction_root = Path(tempfile.mkdtemp(prefix=f"{sequence_id}-", dir=output_root))
    prediction_dir = prediction_root / "prediction"
    prediction_dir.mkdir()
    fasta_path = prediction_root / f"{sequence_id}.fasta"
    fasta_path.write_text(f">{sequence_id}\n{sequence}\n", encoding="utf-8")

    command = [
        sys.executable, "-m", "colabfold.batch",
        str(fasta_path), str(prediction_dir),
        "--model-type", model_type,
        "--msa-mode", msa_mode,
        "--num-models", str(num_models),
        "--num-recycle", str(num_recycles),
        "--num-seeds", "1",
        "--num-relax", "0",
        "--rank", "plddt",
        "--compile-mode", "fast",
        "--max-msa", "64:128",
        "--overwrite-existing-results",
    ]
    if use_templates:
        command.append("--templates")
    if inference_backend == "gpu":
        command.extend(["--use-fast-kernels", "--kernel-backend", "auto"])

    command_environment = os.environ.copy()
    compilation_cache = Path(f"/tmp/colabfold-jax-cache-{inference_backend}")
    compilation_cache.mkdir(parents=True, exist_ok=True)
    command_environment.setdefault("JAX_COMPILATION_CACHE_DIR", str(compilation_cache))
    command_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if inference_backend == "cpu":
        command_environment["JAX_PLATFORMS"] = "cpu"
        command_environment["CUDA_VISIBLE_DEVICES"] = ""

    started_at = time.monotonic()
    subprocess.run(command, env=command_environment, check=True)
    wall_time_seconds = time.monotonic() - started_at

    structure_candidates = sorted(prediction_dir.glob(f"{sequence_id}_relaxed_rank_001_*.pdb"))
    structure_candidates = structure_candidates or sorted(
        prediction_dir.glob(f"{sequence_id}_unrelaxed_rank_001_*.pdb")
    )
    score_candidates = sorted(prediction_dir.glob(f"{sequence_id}_scores_rank_001_*.json"))
    alignment_path = prediction_dir / f"{sequence_id}.a3m"
    if not structure_candidates or not score_candidates or not alignment_path.is_file():
        raise FileNotFoundError(f"Missing structure, scores, or alignment in {prediction_dir}.")

    fold_result = {
        "sequence": sequence,
        "sequence_id": sequence_id,
        "metadata": metadata,
        "model_type": model_type,
        "msa_mode": msa_mode,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "use_templates": use_templates,
        "inference_backend": inference_backend,
        "fasta_path": str(fasta_path),
        "prediction_dir": str(prediction_dir),
        "pred_output_path": str(structure_candidates[0]),
        "score_path": str(score_candidates[0]),
        "alignment_path": str(alignment_path),
        "wall_time_seconds": wall_time_seconds,
    }
    print(f"Prediction complete in {wall_time_seconds:.1f} seconds: {prediction_dir}")
    return (fold_result,)


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 5. Inspect Alignment and Confidence

    The following helpers read the A3M alignment and prediction scores, then
    save plots of alignment coverage, per-residue pLDDT, and predicted aligned
    error (PAE), when available. Lowercase A3M insertion characters are removed
    before calculating coverage.

    pLDDT measures local confidence on a 0–100 scale; PAE describes uncertainty
    in relative residue placement. High local confidence alone does not establish
    biological function or correct domain placement. See the
    [AlphaFold FAQ](https://alphafold.ebi.ac.uk/faq) for interpretation.

    The cells after the helpers display the structure and plots. We use the same
    files when logging Weights & Biases media, so local inspection and the experiment record
    show the same results.
    """)
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
    print(f"Mean pLDDT: {prediction_summary['mean_plddt']:.1f}")
    print(f"Unique MSA sequences: {prediction_summary['msa_sequences']:,}")
    if prediction_summary["ptm"] is not None:
        print(f"pTM: {prediction_summary['ptm']:.2f}")
    return (prediction_summary,)


@app.cell
def _(prediction_summary):
    pdb_text = Path(prediction_summary["pred_output_path"]).read_text(encoding="utf-8")
    structure_view = py3Dmol.view(width=800, height=600)
    structure_view.addModel(pdb_text, "pdb")
    structure_view.setStyle({
        "cartoon": {
            "colorscheme": {"prop": "b", "gradient": "roygb", "min": 0, "max": 100}
        }
    })
    structure_view.zoomTo()
    structure_view
    return


@app.cell
def _(prediction_summary):
    result_plots = [
        mo.image(prediction_summary["msa_plot_path"]),
        mo.image(prediction_summary["confidence_plot_path"]),
    ]
    if prediction_summary["pae_plot_path"]:
        result_plots.append(mo.image(prediction_summary["pae_plot_path"]))
    mo.vstack(result_plots)
    return


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 6. Log the Prediction to Weights & Biases

    The next cell creates `prediction_run` with `wandb.init()`. Its configuration
    records the sequence, source metadata, model settings, and inference backend.

    We use three Weights & Biases media types:

    - `wandb.Molecule()` uploads the PDB structure for interactive 3D viewing.
    - `wandb.Image()` logs the alignment and confidence plots.
    - `wandb.Table()` keeps the sequence, molecule, and plots together in one row.

    `prediction_run.log()` records the media alongside numeric metrics:
    mean pLDDT, unique MSA sequence count, total runtime, and pTM when present.
    The context manager finishes the run after logging.
    """)
    return


@app.cell
def _(entity, prediction_summary, project):
    run_config = {
        "sequence": prediction_summary["sequence"],
        "sequence_length": len(prediction_summary["sequence"]),
        "species": prediction_summary["metadata"]["species"],
        "description": prediction_summary["metadata"]["desc"],
        "reference_url": prediction_summary["metadata"]["url"],
        "model_type": prediction_summary["model_type"],
        "msa_mode": prediction_summary["msa_mode"],
        "num_models": prediction_summary["num_models"],
        "num_recycles": prediction_summary["num_recycles"],
        "use_templates": prediction_summary["use_templates"],
        "inference_backend": prediction_summary["inference_backend"],
        "max_msa": "64:128",
    }
    numeric_metrics = {
        "mean_plddt": prediction_summary["mean_plddt"],
        "msa_sequences": prediction_summary["msa_sequences"],
        "wall_time_seconds": prediction_summary["wall_time_seconds"],
    }
    if prediction_summary["ptm"] is not None:
        numeric_metrics["ptm"] = prediction_summary["ptm"]

    with wandb.init(
        entity=entity,
        project=project,
        name=f"colabfold-{prediction_summary['sequence_id']}",
        config=run_config,
        job_type="protein-structure-prediction",
    ) as prediction_run:
        molecule = wandb.Molecule(prediction_summary["pred_output_path"])
        msa_plot = wandb.Image(prediction_summary["msa_plot_path"])
        confidence_plot = wandb.Image(prediction_summary["confidence_plot_path"])
        pae_plot = (
            wandb.Image(prediction_summary["pae_plot_path"])
            if prediction_summary["pae_plot_path"]
            else None
        )
        prediction_table = wandb.Table(columns=[
            "tag", "species", "sequence", "molecule", "MSA coverage",
            "confidence", "predicted aligned error",
        ])
        prediction_table.add_data(
            prediction_summary["metadata"]["name"],
            prediction_summary["metadata"]["species"],
            prediction_summary["sequence"],
            molecule,
            msa_plot,
            confidence_plot,
            pae_plot,
        )
        log_payload = {
            **numeric_metrics,
            "view_3D": molecule,
            "msa_coverage": msa_plot,
            "plddt_confidence": confidence_plot,
            "predicted_molecules": prediction_table,
        }
        if pae_plot is not None:
            log_payload["predicted_aligned_error"] = pae_plot
        prediction_run.log(log_payload)

    return 


@app.cell(hide_code=True)
def _(run_url):
    mo.md(f"""
    ## 7. Compare Predictions

    Open the Weights & Biases run to inspect its configuration, metrics,
    confidence plots, and 3D molecule. The `predicted_molecules` table connects
    the sequence to its structure and plots.

    Try another sample or increase the number of models. Click **Align and fold**
    again to produce a new prediction and Weights & Biases run, then compare confidence and
    runtime in your project. The wall time includes alignment, model setup, and
    inference; first-run downloads can make it substantially longer.

    The public MSA service is intended for small, serial queries. For larger
    workloads, follow ColabFold's local-database instructions.
    """)
    return


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## References

    Protein-structure predictions require scientific review and are not for
    clinical use. ColabFold, AlphaFold, model parameters, and the MSA service
    have their own licenses and terms; consult the
    [ColabFold documentation](https://github.com/sokrypton/ColabFold) and
    [AlphaFold license information](https://github.com/google-deepmind/alphafold#license-and-disclaimer).

    Publications should cite the
    [ColabFold paper](https://doi.org/10.1038/s41592-022-01488-1) and
    [AlphaFold paper](https://doi.org/10.1038/s41586-021-03819-2).
    """)
    return


if __name__ == "__main__":
    app.run()
