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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure_gpt_medical_notes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell
def _():
    from typing import Dict, List, Literal, Optional, Tuple

    import instructor
    import openai
    import pandas as pd
    import weave
    from pydantic import BaseModel, Field
    from set_env import set_env
    import json
    import asyncio

    return Dict, Tuple, asyncio, json, openai, pd, set_env, weave


@app.cell
def _(set_env):
    set_env("OPENAI_API_KEY")
    set_env("WANDB_API_KEY")
    set_env("AZURE_OPENAI_ENDPOINT")
    set_env("AZURE_OPENAI_API_KEY")
    print("Env set")
    return


@app.cell
def _():
    from utils.config import ENTITY, WEAVE_PROJECT

    return ENTITY, WEAVE_PROJECT


@app.cell
def _(ENTITY, WEAVE_PROJECT, weave):
    weave.init(f"{ENTITY}/{WEAVE_PROJECT}")
    return


@app.cell
def _():
    N_SAMPLES = 67
    return (N_SAMPLES,)


@app.cell
def _(openai):
    client = openai.OpenAI()
    return (client,)


@app.cell
def _(N_SAMPLES, Tuple, pd):
    def load_medical_data(url: str, num_samples: int = N_SAMPLES) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load medical data and split into train and test sets
    
        Args:
            url: URL of the CSV file
            num_samples: Number of samples to load
        
        Returns:
            Tuple of (train_df, test_df)
        """
        df = pd.read_csv(url)
        df = df.sample(n=num_samples, random_state=42)  # Sample and shuffle data
    
        # Split into 80% train, 20% test
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]
    
        return train_df, test_df

    return (load_medical_data,)


@app.cell
def _():
    medical_dataset_url = "https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/train.csv"
    return (medical_dataset_url,)


@app.cell
def _(load_medical_data, medical_dataset_url):
    train_df, test_df = load_medical_data(medical_dataset_url)
    train_samples = train_df.to_dict("records")
    test_samples = test_df.to_dict("records")
    return test_df, test_samples, train_df, train_samples


@app.cell
def _(train_samples):
    train_samples[0]
    return


@app.cell
def _(test_samples):
    test_samples[0]
    return


@app.cell
def _(json, pd):
    def convert_to_jsonl(df: pd.DataFrame, output_file: str = "medical_conversations.jsonl"):
        """
        Convert medical dataset to JSONL format with conversation structure
    
        Args:
            df: DataFrame to convert
            output_file: Output JSONL filename
        """
    
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Create the conversation structure
                conversation = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a medical scribe assistant. Your task is to accurately document medical conversations between doctors and patients, creating detailed medical notes that capture all relevant clinical information."
                        },
                        {
                            "role": "user",
                            "content": row['dialogue']
                        },
                        {
                            "role": "assistant",
                            "content": row['note']
                        }
                    ]
                }
            
                # Write as JSON line
                json_line = json.dumps(conversation, ensure_ascii=False)
                f.write(json_line + '\n')
    
        print(f"Converted {len(df)} records to {output_file}")

    return (convert_to_jsonl,)


@app.cell
def _(convert_to_jsonl, test_df, train_df):
    convert_to_jsonl(train_df, "medical_conversations_train.jsonl")
    convert_to_jsonl(test_df, "medical_conversations_test.jsonl")
    return


@app.cell
def _():
    from utils.prompts import medical_task, medical_system_prompt

    return medical_system_prompt, medical_task


@app.cell
def _(Dict, client, medical_system_prompt, medical_task, weave):
    def format_dialogue(dialogue: str):
        dialogue = dialogue.replace("\n", " ")
        transcript = f"Dialogue: {dialogue}"
        return transcript


    @weave.op()
    def process_medical_record(dialogue: str) -> Dict:
        transcript = format_dialogue(dialogue)
        prompt = medical_task.format(transcript=transcript)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": medical_system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        extracted_info = response.choices[0].message.content

        return {
            "input": transcript,
            "output": extracted_info,
        }

    return (process_medical_record,)


@app.cell
def _(client, json, weave):
    # Define the LLM scoring function
    @weave.op()
    async def medical_note_accuracy(note: str, output: dict) -> dict:
        scoring_prompt = """Compare the generated medical note with the ground truth note and evaluate accuracy.
        Score as 1 if the generated note captures the key medical information accurately, 0 if not.
        Output in valid JSON format with just a "score" field.
    
        Ground Truth Note:
        {ground_truth}
    
        Generated Note:
        {generated}"""
    
        prompt = scoring_prompt.format(
            ground_truth=note,
            generated=output['output']
        )
    
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)

    return (medical_note_accuracy,)


@app.cell
def _(medical_note_accuracy, test_samples, weave):
    # Create evaluation for test samples
    test_evaluation = weave.Evaluation(
        name='medical_record_extraction_test',
        dataset=test_samples,
        scorers=[medical_note_accuracy]
    )
    return (test_evaluation,)


@app.cell
def _():
    try:
        in_jupyter = True
    except ImportError:
        in_jupyter = False
    if in_jupyter:
        import nest_asyncio

        nest_asyncio.apply()
    return


@app.cell
def _(asyncio, process_medical_record, test_evaluation):
    test_results = asyncio.run(test_evaluation.evaluate(process_medical_record))
    print(f"Completed test evaluation")
    return


@app.cell
def _(Dict, weave):
    import os
    from openai import AzureOpenAI

    # Initialize Azure client
    azure_client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01"
    )

    @weave.op()
    def process_medical_record_azure(dialogue: str) -> Dict:

        response = azure_client.chat.completions.create(
            model="gpt-35-turbo-0125-ft-d30b3aee14864c29acd9ac54eb92457f",
            messages=[
                {"role": "system", "content": "You are a medical scribe assistant. Your task is to accurately document medical conversations between doctors and patients, creating detailed medical notes that capture all relevant clinical information."},
                {"role": "user", "content": dialogue},
            ],
        )

        extracted_info = response.choices[0].message.content

        return {
            "input": dialogue,
            "output": extracted_info,
        }

    return (process_medical_record_azure,)


@app.cell
def _(asyncio, process_medical_record_azure, test_evaluation):
    test_results_azure = asyncio.run(test_evaluation.evaluate(process_medical_record_azure))
    print(f"Completed test evaluation")
    return


if __name__ == "__main__":
    app.run()
