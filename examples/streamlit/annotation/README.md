# Streamlit + W&B for LLM Annotation

Try W&B Tables and Streamlit data editor to annotate data for LLMs. 

This repo builds a simple app for labeling and annotating tables. These tables can be sample text outputs from developing LLMs. For example, you can test a set of sample prompts, get the LLM responses, and then save those results to be annotated in this workflow.

Here's a quick overview:
1. Set up a virtual environment and install dependencies
2. Run the Streamlit App locally, and customize the UI
3. Annotate and version tables of data using W&B + Streamlit

## 1. Set Up Environment
Start your virtual environment, and install all dependencies from the requirements file. This simple app just uses `pandas` and `streamlit`.
```shell
pip install -r requirements.txt
```

## 2. Run the Streamlit App
Open the `wandb_streamlit_app.py` file, take a look at the app definition, and make edits.
Edit the column configuration, as defined [in the Streamlit docs.](https://docs.streamlit.io/library/api-reference/data/st.column_config)

### Optional: Apply a custom theme

Create a hidden `.streamlit` folder in the root of the project with the following command:
```shell
mkdir .streamlit
```

Copy the `config.toml` file to the hidden folder and edit as desired. This will apply that theme to the Streamlit app as you run it. There are multiple ways to work with custom themes but this is one of the simplest ways.
- [This video tutorial](https://www.youtube.com/watch?v=Mz12mlwzbVU) provides a nice walkthrough of creating and applying custom themes
- W&B color palette detail can be found [here](https://congenial-broccoli-daa12ae2.pages.github.io/?path=%2Fstory%2Fcommon-colors--overview)

Finally, run the Streamlit app on localhost:
```shell
streamlit run wandb_streamlit_app.py
```
From there, you can edit the columns that have been configured for labeling:
![image](https://github.com/wandb/examples/assets/6355078/eff3ad41-4550-42e7-92ae-00ba5d46f5b5)


## 3. Annotate and Version Tables

This is what the data looks like from the CSV of sample LLM inputs and outputs:
![image](https://github.com/wandb/examples/assets/6355078/6a20a3cd-3ebd-4372-bb79-64c54f965d55)


Once loaded as a W&B Table, we have a clean, annotated version of this workflow, complete with metadata, in our system of record:
![image](https://github.com/wandb/examples/assets/6355078/4cd1357a-71c3-4c06-bd55-caf1042e45f0)



