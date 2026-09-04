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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/llamaindex/llamaindex_local_models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **TL;DR:** Build a RAG application using llamaindex and local models (embedding + LLM), with [weave](https://wandb.github.io/weave/) for LLM observability
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📦 Packages and Basic Setup
    ---
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !wget https://controlroom.jurassicoutpost.com/app/uploads/2016/05/JurassicPark-Final.pdf
    # !pip install -qU llama-index-callbacks-wandb
    # !pip install -qU llama-index-llms-huggingface
    # !pip install -qU llama-index-readers-file pymupdf
    # !pip install -qU llama-index-embeddings-huggingface
    # !pip install -qU weave ml-collections accelerate
    return


@app.cell
def _():
    import wandb
    import weave
    from llama_index.callbacks.wandb import WandbCallbackHandler

    wandb.login()
    weave.init("llamaindex-weave-jurassic-qna")
    wandb_callback = WandbCallbackHandler(
        run_args={"project": "llamaindex-weave-jurassic-qna"}
    )
    return (wandb_callback,)


@app.cell
def _():
    # @title ⚙️ Configuration
    import ml_collections

    from llama_index.core import Settings


    def get_config() -> ml_collections.ConfigDict:
        config = ml_collections.ConfigDict()
        config.model: str = "Writer/camel-5b-hf"  # @param {type: "string"}
        config.embedding_model: str = "BAAI/bge-small-en-v1.5"  # @param {type: "string"}
        config.fetch_index_from_wandb: bool = True  # @param {type: "boolean"}
        config.wandb_entity: str = "sauravmaheshkar"  # @param {type: "string"}

        return config


    config = get_config()
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💿 The Dataset
    ---

    In this example, we'll use the original Jurassic Park screenplay to act as our dataset.
    """)
    return


@app.cell
def _():
    from llama_index.core import Document
    from llama_index.readers.file import PyMuPDFReader

    documents = PyMuPDFReader().load(
        file_path="/content/JurassicPark-Final.pdf", metadata=True
    )

    doc_text = "\n\n".join([d.get_content() for d in documents])
    docs = [Document(text=doc_text)]
    return (documents,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✍️ Model Architecture & Training
    ---

    Since we're using all local models in this example, we'll have to our own Embedding model and llm. In this particular example we'll use "`BAAI/bge-small-en-v1.5`" as our local embedding model and "`Writer/camel-5b-hf`" as the local LLM.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # 
    # Settings.embed_model = HuggingFaceEmbedding(model_name=config.embedding_model)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # import torch
    # from llama_index.core import PromptTemplate
    # from llama_index.llms.huggingface import HuggingFaceLLM
    # 
    # query_wrapper_prompt = PromptTemplate(
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{query_str}\n\n### Response:"
    # )
    # 
    # Settings.llm = HuggingFaceLLM(
    #     context_window=2048,
    #     max_new_tokens=256,
    #     generate_kwargs={"do_sample": False},
    #     query_wrapper_prompt=query_wrapper_prompt,
    #     tokenizer_name=config.model,
    #     model_name=config.model,
    #     device_map="auto",
    #     tokenizer_kwargs={"max_length": 2048},
    #     model_kwargs={"torch_dtype": torch.float16},
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🗂 Creating a Index
    ---

    Based on the value you set for `config.fetch_index_from_wandb` we can either create our own index, or simply download the index stored as an artifact.
    """)
    return


@app.cell
def _(config, documents, wandb_callback):
    from llama_index.core import VectorStoreIndex

    if not config.fetch_index_from_wandb:
        index = VectorStoreIndex.from_documents(documents)
        wandb_callback.persist_index(index, index_name="camel-5b-hf-index")
    return


@app.cell
def _(config, wandb_callback):
    from llama_index.core import load_index_from_storage
    if config.fetch_index_from_wandb:
        storage_context = wandb_callback.load_storage_context(artifact_url='sauravmaheshkar/llamaindex-local-models-index/camel-5b-hf-index:v0')
        index_1 = load_index_from_storage(storage_context)  # Load the index and initialize a query engine
    return (index_1,)


@app.cell
def _(index_1):
    query_engine = index_1.as_query_engine()
    response = query_engine.query('Are Velociraptors pack hunters ?')
    print(response, sep='\n')
    return


@app.cell
def _(wandb_callback):
    wandb_callback.finish()
    return


if __name__ == "__main__":
    app.run()
