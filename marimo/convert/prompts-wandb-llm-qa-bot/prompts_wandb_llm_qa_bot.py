# /// script
# dependencies = ["chromadb", "langchain", "openai", "pytube", "tiktoken", "wandb", "youtube-transcript-api"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_LLM_QA_bot.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{prompt-qa-bot} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building an LLM App for Document Retrieval / Extraction
    <!--- @wandbcode{prompt-qa-bot} -->
    This tutorial runs through [this report](https://wandb.ai/gladiator/gradient_dissent_qabot/reports/Building-a-Q-A-Bot-for-Weights-Biases-Gradient-Dissent-Podcast--Vmlldzo0MTcyMDQz) on how to build a basic LLM App for retrieval-augmented question-answering.
    - Track datasets and embeddings as artifacts
    - Track prompts and chain executions
    - Log token counts and cost
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb langchain pytube tiktoken openai youtube-transcript-api chromadb !pip install -qqq wandb langchain pytube tiktoken openai youtube-transcript-api chromadb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set up OpenAI API Key
    """)
    return


@app.cell
def _():
    from getpass import getpass
    import os

    if os.getenv("OPENAI_API_KEY") is None:
      if any(['VSCODE' in x for x in os.environ.keys()]):
        print('Please enter password in the VS Code prompt at the top of your VS Code window!')
      os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")

    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
    print("OpenAI API key configured")
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Set up config and environment variables
    - NOTE: set the `entity` to your username or team name
    - Set wandb [environment variables](https://docs.wandb.ai/guides/track/environment-variables) to change behavior of logging
    - `ENTITY` - username or team where your projects live
    - `PROJECT` - project where your runs will live
    - `LANGCHAIN_WANDB_TRACING` - automatically logs langchain traces, inputs and outputs as part of runs in Weights and Biases
    """)
    return


@app.cell
def _(os):
    from dataclasses import dataclass
    from pathlib import Path
    project_name = 'gradient-dissent-qabot'
    entity = 'wandb'
    TOTAL_EPISODES = 5  #@param
    playlist_url = 'https://www.youtube.com/playlist?list=PLD80i8An1OEEb1jP0sjEyiLG8ULRXFob_'  #@param
    root_data_dir = Path('/contents/data')
    root_artifact_dir = Path('downloaded_artifacts')
    yt_podcast_data_artifact = f'{entity}/{project_name}/yt_podcast_transcript:latest'
    summarized_data_artifact = f'{entity}/{project_name}/summarized_podcasts:latest'
    summarized_que_data_artifact = f'{entity}/{project_name}/summarized_que_podcasts:latest'
    transcript_embeddings_artifact = f'{entity}/{project_name}/transcript_embeddings:latest'
    os.makedirs('/contents/data', exist_ok=True)
    os.environ['LANGCHAIN_WANDB_TRACING'] = 'true'
    os.environ['WANDB_PROJECT'] = project_name
    os.environ['WANDB_ENTITY'] = entity
    return (
        TOTAL_EPISODES,
        entity,
        playlist_url,
        project_name,
        root_artifact_dir,
        root_data_dir,
        summarized_data_artifact,
        transcript_embeddings_artifact,
        yt_podcast_data_artifact,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log in to W&B
    - You can explicitly login using `wandb login` or `wandb.login()` (See below)
    - Alternatively you can set environment variables. There are several env variables which you can set to change the behavior of W&B logging. The most important are:
        - `WANDB_API_KEY` - create a new API key in your "Settings" section under your profile at [wandb.ai/settings](https://wandb.ai/settings)
        - `WANDB_BASE_URL` - this is the url of the W&B server (You only need this if you are using a private instance)
    - Create a new API key in "Profile" -> "Settings" in the W&B App. Store your API key securely. It can only be viewed once when created.

    ![api_token](https://drive.google.com/uc?export=view&id=1Xn7hnn0rfPu_EW0A_-32oCXqDmpA0-kx)
    """)
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
    import time
    import pandas as pd
    from langchain.document_loaders import YoutubeLoader
    from pytube import Playlist, YouTube
    from tqdm import tqdm

    def retry_access_yt_object(url, max_retries=5, interval_secs=5):
        """
        Retries creating a YouTube object with the given URL and accessing its title several times
        with a given interval in seconds, until it succeeds or the maximum number of attempts is reached.
        If the object still cannot be created or the title cannot be accessed after the maximum number
        of attempts, the last exception is raised.
        """
        last_exception = None
        for i in range(max_retries):
            try:
                yt = YouTube(url)
                title = yt.title
                return yt
            except Exception as err:  # Access the title of the YouTube object.
                last_exception = err  # Return the YouTube object if successful.
                print(f'Failed to create YouTube object or access title. Retrying... ({i + 1}/{max_retries})')
                time.sleep(interval_secs)  # Keep track of the last exception raised.
        raise last_exception  # Wait for the specified interval before retrying.  # If the YouTube object still cannot be created or the title cannot be accessed after the maximum number of attempts, raise the last exception.

    return Playlist, YoutubeLoader, pd, retry_access_yt_object, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log Data Snapshots as Artifacts

    W&B is very unopinionated with regard to how you track your experiments.  We could log data in any number of ways.
    * Log one artifact which represents all the data - training, validation, and test data to one artifact
    * Log several artifacts - one for each of the training, validation, and test data loaders.

    It is a matter of what best suites your needs and workflows and expectations.

    ### Anatomy of an artifact

    The `Artifact` class will correspond to an entry in the W&B Artifact registry.  The artifact has
    * a name
    * a type
    * metadata
    * description
    * files, directory of files, or references

    Example usage
    ```
    run = wandb.init(project = "my-project")
    artifact = wandb.Artifact(name = "my_artifact", type = "data")
    artifact.add_file("/path/to/my/file.txt")
    run.log_artifact(artifact)
    run.finish()
    ```
    """)
    return


@app.cell
def _(
    Playlist,
    TOTAL_EPISODES,
    YoutubeLoader,
    entity,
    pd,
    playlist_url,
    project_name,
    retry_access_yt_object,
    root_data_dir,
    tqdm,
    wandb,
):
    run = wandb.init(project=project_name, entity=entity, job_type='dataset')
    playlist = Playlist(playlist_url)
    playlist_video_urls = playlist.video_urls[0:TOTAL_EPISODES]
    print(f'There are total {len(playlist_video_urls)} videos in the playlist.')
    video_data = []
    for video in tqdm(playlist_video_urls, total=len(playlist_video_urls)):
        try:
            curr_video_data = {}
            yt = retry_access_yt_object(video, max_retries=25, interval_secs=2)
            curr_video_data['title'] = yt.title
            curr_video_data['url'] = video
            curr_video_data['duration'] = yt.length
            curr_video_data['publish_date'] = yt.publish_date.strftime('%Y-%m-%d')
            loader = YoutubeLoader.from_youtube_url(video)
            transcript = loader.load()[0].page_content
            transcript = ' '.join(transcript.split())
            curr_video_data['transcript'] = transcript
            curr_video_data['total_words'] = len(transcript.split())
            video_data.append(curr_video_data)
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print(f'Failed to scrape {video}')  # the exception type
    print(f'Total podcast episodes scraped: {len(video_data)}')  # arguments stored in .args
    df = pd.DataFrame(video_data)
    data_path = root_data_dir / 'yt_podcast_transcript.csv'
    df.to_csv(data_path, index=False)
    _artifact = wandb.Artifact('yt_podcast_transcript', type='dataset')
    _artifact.add_file(data_path)
    # save the scraped data to a csv file
    # upload the scraped data to wandb
    run.log_artifact(_artifact)
    return df, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Log a wandb Table to interact with your data
    - Here we log the dataframe of metadata about the youtube transcripts (urls, length, transcripts)
    - This allows us to interrogate the original data (filtering, grouping, etc.)
    """)
    return


@app.cell
def _(df, run, wandb):
    # create wandb table
    _table = wandb.Table(dataframe=df)
    run.log({'yt_podcast_transcript': _table})
    run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summarize YouTube Transcripts
    - Here we summarize the transcripts in chunks, summarizing each chunk and then summarizing the summaries using the LangChain `load_summarize_chain`
    - We can do this in parallel since each chunk of a transcript can be summarized independently so we employ `map_reduce`
    """)
    return


@app.cell
def _(os, pd, root_artifact_dir, wandb):
    from langchain.callbacks import get_openai_callback
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chat_models import ChatOpenAI
    from langchain.document_loaders import DataFrameLoader
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import TokenTextSplitter

    def get_data(artifact_name: str, total_episodes: int=None):
        podcast_artifact = wandb.use_artifact(artifact_name)
        podcast_artifact_dir = podcast_artifact.download(root_artifact_dir)
        filename = artifact_name.split(':')[0].split('/')[-1]
        df = pd.read_csv(os.path.join(podcast_artifact_dir, f'{filename}.csv'))
        if total_episodes is not None:
            df = df.iloc[:total_episodes]
        return df

    def summarize_episode(episode_df: pd.DataFrame):
        loader = DataFrameLoader(episode_df, page_content_column='transcript')
        data = loader.load()
        text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        print(f"Number of documents for podcast {data[0].metadata['title']}: {len(docs)}")
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
        map_prompt = "Write a concise summary of the following short transcript from a podcast.\n    Don't add your opinions or interpretations.\n\n    {text}\n\n    CONCISE SUMMARY:"
        combine_prompt = 'You have been provided with summaries of chunks of transcripts from a podcast.\n    Your task is to merge these intermediate summaries to create a brief and comprehensive summary of the entire podcast.\n    The summary should encompass all the crucial points of the podcast.\n    Ensure that the summary is atleast 2 paragraph long and effectively captures the essence of the podcast.\n    {text}\n\n    SUMMARY:'  # load docs into langchain format
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=['text'])
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['text'])
        chain = load_summarize_chain(llm, chain_type='map_reduce', return_intermediate_steps=True, map_prompt=map_prompt_template, combine_prompt=combine_prompt_template)
        summary = chain({'input_documents': docs})  # split the documents
        return summary  # initialize LLM  # define map prompt  # define combine prompt  # initialize the summarizer chain

    return (
        ChatOpenAI,
        DataFrameLoader,
        PromptTemplate,
        TokenTextSplitter,
        get_data,
        get_openai_callback,
        summarize_episode,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Execute Summary Chain and log results
    - You can instantiate a `WandbTracer` and pass in additional config about this LangChain run.
    - Log the outputs of the chain like tokens used, cost, etc.
    - Log the resulting summaries as artifacts
    """)
    return


@app.cell
def _(
    TOTAL_EPISODES,
    get_data,
    get_openai_callback,
    os,
    root_data_dir,
    summarize_episode,
    tqdm,
    wandb,
    yt_podcast_data_artifact,
):
    from langchain.callbacks.tracers import WandbTracer
    _tracer = WandbTracer(run_args={'job_type': 'summarize'})
    df_1 = get_data(artifact_name=yt_podcast_data_artifact, total_episodes=TOTAL_EPISODES)
    summaries = []
    with get_openai_callback() as _cb:
        for _episode in tqdm(df_1.iterrows(), total=len(df_1), desc='Summarizing episodes'):
            _episode_data = _episode[1].to_frame().T
            summary = summarize_episode(_episode_data)
            summaries.append(summary['output_text'])
        print('*' * 25)
        print(_cb)
        print('*' * 25)
        wandb.log({'total_prompt_tokens': _cb.prompt_tokens, 'total_completion_tokens': _cb.completion_tokens, 'total_tokens': _cb.total_tokens, 'total_cost': _cb.total_cost})
    df_1['summary'] = summaries
    path_to_save = os.path.join(root_data_dir, 'summarized_podcasts.csv')
    df_1.to_csv(path_to_save, index=False)
    _artifact = wandb.Artifact('summarized_podcasts', type='dataset')
    _artifact.add_file(path_to_save)
    wandb.log_artifact(_artifact)
    _table = wandb.Table(dataframe=df_1)
    wandb.log({'summarized_podcasts': _table})
    _tracer.finish()
    return (WandbTracer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embed the contents of the YouTube transcripts
    - Here we use OpenAI embeddings and [ChromaDB](https://www.trychroma.com/) to embed the summaries to make them queriable via vector similarity search when we ask contextual questions to the LLM
    - Use `wandb.log` and artifacts to log the resulting ChromaDB serialized embeddings.
    """)
    return


@app.cell
def _(
    DataFrameLoader,
    TokenTextSplitter,
    os,
    pd,
    root_artifact_dir,
    root_data_dir,
    wandb,
):
    from dataclasses import asdict
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    def get_data_1(artifact_name: str, total_episodes=None):
        podcast_artifact = wandb.use_artifact(artifact_name, type='dataset')
        podcast_artifact_dir = podcast_artifact.download(root_artifact_dir)
        filename = artifact_name.split(':')[0].split('/')[-1]
        df = pd.read_csv(os.path.join(podcast_artifact_dir, f'{filename}.csv'))
        if total_episodes is not None:
            df = df.iloc[:total_episodes]
        return df

    def create_embeddings(episode_df: pd.DataFrame, index: int):
        loader = DataFrameLoader(episode_df, page_content_column='transcript')
        data = loader.load()
        text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)
        title = data[0].metadata['title']
        print(f'Number of documents for podcast {title}: {len(docs)}')
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(docs, embeddings, persist_directory=os.path.join(root_data_dir / 'chromadb', str(index)))
        db.persist()  # load docs into langchain format  # split the documents  # initialize embedding engine

    return Chroma, OpenAIEmbeddings, create_embeddings, get_data_1


@app.cell
def _(
    TOTAL_EPISODES,
    WandbTracer,
    create_embeddings,
    get_data_1,
    get_openai_callback,
    root_data_dir,
    summarized_data_artifact,
    tqdm,
    wandb,
):
    _tracer = WandbTracer(run_args={'job_type': 'embed_transcripts'})
    df_2 = get_data_1(artifact_name=summarized_data_artifact, total_episodes=TOTAL_EPISODES)
    with get_openai_callback() as _cb:
        for _episode in tqdm(df_2.iterrows(), total=len(df_2), desc='Embedding transcripts'):
            _episode_data = _episode[1].to_frame().T
            create_embeddings(_episode_data, index=_episode[0])
        print('*' * 25)
        print(_cb)
        print('*' * 25)
        wandb.log({'total_prompt_tokens': _cb.prompt_tokens, 'total_completion_tokens': _cb.completion_tokens, 'total_tokens': _cb.total_tokens, 'total_cost': _cb.total_cost})
    _artifact = wandb.Artifact('transcript_embeddings', type='dataset')
    _artifact.add_dir(root_data_dir / 'chromadb')
    wandb.log_artifact(_artifact)
    _tracer.finish()
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ask Questions Against your Summarized Documents

    Finally we tie everything together:
    1. We can pull down our ChromaDB embeddings from W&B
    2. Pass them along with a prompt template for QA to the `RetrievalQA` chain and start asking questions!
    """)
    return


@app.cell
def _(
    ChatOpenAI,
    Chroma,
    OpenAIEmbeddings,
    PromptTemplate,
    chromadb_dir,
    df_2,
    get_openai_callback,
    os,
):
    from langchain.chains import RetrievalQA

    def get_answer(podcast: str, question: str):
        index = df_2[df_2['title'] == podcast].index[0]
        db_dir = os.path.join(chromadb_dir, str(index))
        embeddings = OpenAIEmbeddings()
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        prompt_template = "Use the following pieces of context to answer the question.\n  If you don't know the answer, just say that you don't know, don't try to make up an answer.\n  Don't add your opinions or interpretations. Ensure that you complete the answer.\n  If the question is not relevant to the context, just say that it is not relevant.\n\n  CONTEXT:\n  {context}\n\n  QUESTION: {question}\n\n  ANSWER:"
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        retriever = db.as_retriever()
        retriever.search_kwargs['k'] = 2
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), chain_type='stuff', retriever=retriever, chain_type_kwargs={'prompt': prompt}, return_source_documents=True)
        with get_openai_callback() as _cb:
            result = qa({'query': question})
            print(_cb)
        answer = result['result']
        return answer

    return (get_answer,)


@app.cell
def _(
    pd,
    root_data_dir,
    summarized_data_artifact,
    transcript_embeddings_artifact,
    wandb,
):
    # download and read data
    api = wandb.Api()
    artifact_df = api.artifact(summarized_data_artifact)
    artifact_df.download(root_data_dir)
    artifact_embeddings = api.artifact(transcript_embeddings_artifact)
    chromadb_dir = artifact_embeddings.download(root_data_dir / 'chromadb')
    df_path = root_data_dir / 'summarized_podcasts.csv'
    df_3 = pd.read_csv(df_path)
    return chromadb_dir, df_3


@app.cell
def _(TOTAL_EPISODES, df_3):
    df_3['title'].tolist()[0:TOTAL_EPISODES]
    return


@app.cell
def _(WandbTracer, get_answer):
    _tracer = WandbTracer(run_args={'job_type': 'retriealQA'})
    answer = get_answer('Enabling LLM-Powered Applications with Harrison Chase of LangChain', 'What did Harrison Chase say?')
    print(answer)
    _tracer.finish()
    return


if __name__ == "__main__":
    app.run()
