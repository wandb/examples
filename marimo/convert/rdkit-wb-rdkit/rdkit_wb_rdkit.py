# /// script
# dependencies = ["rdkit-pypi", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/rdkit/wb_rdkit.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{rdkit_molecules} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{rdkit_molecules} -->

    # Logging RDKit Molecular Data

    [RDKit](https://www.rdkit.org/) is a popular open source toolkit for cheminformatics. In version `0.12.7` of the `wandb` client library, we added `wandb.Molecule` support for `rdkit` data formats. In particular, you can now initialize `wandb.Molecule` from [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) strings, [`rdkit.Chem.rdchem.Mol`](https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Mol) objects, and files in `rdkit`-supported formats, such as `.mol`.

    This Colab showcases how you can log `rdkit` molecular data in Weights & Biases and visualize it both in 3D and 2D.

    ###[Click here](https://wandb.ai/anmolmann/rdkit_molecules) to view and interact with a live W&B Dashboard built with this notebook.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    # packages added via marimo's package management: rdkit-pypi !pip install rdkit-pypi -qqq
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview
    In this example, we're using Google Colab as a convenient hosted environment, but you can run your own training scripts from anywhere and visualize metrics and data with W&B's experiment tracking tool.

    As an example, we will initialize `wandb.Molecule` objects from different `rdkit` formats and log them to a `wandb.Table` for visualization.
    """)
    return


@app.cell
def _():
    import datetime
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw

    return Chem, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us save a `.mol` file:
    """)
    return


@app.cell
def _(Chem):
    resveratrol = Chem.MolFromSmiles("Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O")
    Chem.MolToMolFile(resveratrol, "resveratrol.mol")
    return (resveratrol,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D Views of a Molecule
    First, we'll log 2D views of molecule using the [`wandb.Image`](https://docs.wandb.ai/ref/python/data-types/image) data type.
    """)
    return


@app.cell
def _(Chem):
    def mol_to_pil_image(molecule: Chem.rdchem.Mol, width: int = 300, height: int = 300) -> "PIL.Image":
        Chem.AllChem.Compute2DCoords(molecule)
        Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
        pil_image = Chem.Draw.MolToImage(molecule, size=(width, height))
        return pil_image

    return (mol_to_pil_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3D Representations of Molecules
    Now, let us log 3D representations of a few sample molecules using a [`wandb.Table`](https://docs.wandb.ai/ref/python/data-types/table).
    """)
    return


@app.cell
def _(Chem, mol_to_pil_image, resveratrol, wandb):
    smiles = {
        "resveratrol": "Oc1ccc(cc1)C=Cc1cc(O)cc(c1)O",
        "ciprofloxacin": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
        "acetic acid": "CC(=O)O",
    }

    acetic_acid = Chem.MolFromSmiles(smiles["acetic acid"])
    ciprofloxacin = Chem.MolFromSmiles(smiles["ciprofloxacin"])

    data = [
        {
            "name": "resveratrol",
            "smiles": smiles["resveratrol"],
            # wandb.Molecule from a .mol file:
            "molecule": wandb.Molecule.from_rdkit("resveratrol.mol"),
            "molecule_2D": wandb.Image(mol_to_pil_image(resveratrol))
        },
        {
            "name": "ciprofloxacin",
            "smiles": smiles["ciprofloxacin"],
            # wandb.Molecule from a SMILES string:
            "molecule": wandb.Molecule.from_smiles(smiles["ciprofloxacin"]),
            "molecule_2D": wandb.Image(mol_to_pil_image(ciprofloxacin))
        },
        {
            "name": "acetic acid",
            "smiles": smiles["acetic acid"],
            # wandb.Molecule from an rdkit.Chem.rdchem.Mol object:
            "molecule": wandb.Molecule.from_rdkit(acetic_acid),
            "molecule_2D": wandb.Image(mol_to_pil_image(acetic_acid))
        },
    ]
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log Molecular Data to W&B
    """)
    return


@app.cell
def _(data, pd, wandb):
    run = wandb.init(project="rdkit_molecules")

    dataframe = pd.DataFrame.from_records(data)
    table = wandb.Table(dataframe=dataframe)
    wandb.log(
        {
            "table": table,
            "molecules": [substance.get("molecule") for substance in data],
        }
    )

    run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This will produce the following visualization:
    ![Kapture 2021-12-01 at 22 06 37](https://user-images.githubusercontent.com/7557205/144367246-cc052e58-ede4-4374-9307-4f185328c361.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # More about Weights & Biases
    We're always free for academics and open source projects. Email carey@wandb.com with any questions or feature suggestions. Here are some more resources:

    1. [Documentation](http://docs.wandb.com) - Python docs
    2. [Gallery](https://app.wandb.ai/gallery) - example reports in W&B
    3. [Articles](https://www.wandb.com/articles) - blog posts and tutorials
    4. [Community](wandb.me/slack) - join our Slack community forum

    [Sign up or login](https://wandb.ai/login) to W&B to see and interact with your experiments in the browser.
    """)
    return


if __name__ == "__main__":
    app.run()
