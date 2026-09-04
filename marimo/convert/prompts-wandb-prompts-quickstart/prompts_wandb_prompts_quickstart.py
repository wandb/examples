# /// script
# dependencies = ["langchain", "openai", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{prompts-quickstart} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{prompts-quickstart} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **[Weights & Biases Prompts](https://docs.wandb.ai/guides/prompts?utm_source=code&utm_medium=colab&utm_campaign=prompts)** is a suite of LLMOps tools built for the development of LLM-powered applications.

    Use W&B Prompts to visualize and inspect the execution flow of your LLMs, analyze the inputs and outputs of your LLMs, view the intermediate results and securely store and manage your prompts and LLM chain configurations.

    #### [🪄 View Prompts In Action](https://wandb.ai/timssweeney/prompts-demo/)

    **In this notebook we will demostrate W&B Prompts:**

    - Using our 1-line LangChain integration
    - Using our Trace class when building your own LLM Pipelines

    See here for the full [W&B Prompts documentation](https://docs.wandb.ai/guides/prompts)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installation
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb>=0.15.4 !pip install "wandb>=0.15.4" -qqq
    # packages added via marimo's package management: langchain>=0.0.218 openai !pip install "langchain>=0.0.218" openai -qqq
    return


@app.cell
def _():
    import langchain
    assert langchain.__version__ >= "0.0.218", "Please ensure you are using LangChain v0.0.188 or higher"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    This demo requires that you have an [OpenAI key](https://platform.openai.com)
    """)
    return


@app.cell
def _():
    import os
    from getpass import getpass

    if os.getenv("OPENAI_API_KEY") is None:
      os.environ["OPENAI_API_KEY"] = getpass("Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
    assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
    print("OpenAI API key configured")
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B Prompts

    W&B Prompts consists of three main components:

    **Trace table**: Overview of the inputs and outputs of a chain.

    **Trace timeline**: Displays the execution flow of the chain and is color-coded according to component types.

    **Model architecture**: View details about the structure of the chain and the parameters used to initialize each component of the chain.

    After running this section, you will see a new panel automatically created in your workspace, showing each execution, the trace, and the model architecture
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://raw.githubusercontent.com/wandb/examples/master/colabs/prompts/prompts.png" alt="Weights & Biases Prompts image" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maths with LangChain
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set the `LANGCHAIN_WANDB_TRACING` environment variable as well as any other relevant [W&B environment variables](https://docs.wandb.ai/guides/track/environment-variables). This could includes a W&B project name, team name, and more. See [wandb.init](https://docs.wandb.ai/ref/python/init) for a full list of arguments.
    """)
    return


@app.cell
def _(os):
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    os.environ["WANDB_PROJECT"] = "langchain-testing"
    return


@app.cell
def _():
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import load_tools, initialize_agent, AgentType

    return AgentType, ChatOpenAI, initialize_agent, load_tools


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a standard math Agent using LangChain
    """)
    return


@app.cell
def _(AgentType, ChatOpenAI, initialize_agent, load_tools):
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math"], llm=llm)
    math_agent = initialize_agent(tools, 
                                  llm, 
                                  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    return (math_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use LangChain as normal by calling your Agent.

    You will see a Weights & Biases run start and you will be prompted to create a new API key at [wandb.ai/settings](https://wandb.ai/settings) if you haven't already. Store your API key securely. It can only be viewed once when created. Once you enter your API key, the inputs and outputs of your Agent calls will start to be streamed to the Weights & Biases App.
    """)
    return


@app.cell
def _(math_agent):
    # some sample maths questions
    questions = [
      "Find the square root of 5.4.",
      "What is 3 divided by 7.34 raised to the power of pi?",
      "What is the sin of 0.47 radians, divided by the cube root of 27?"
    ]

    for question in questions:
      try:
        # call your Agent as normal
        answer = math_agent.run(question)
        print(answer)
      except Exception as e:
        # any errors will be also logged to Weights & Biases
        print(e)
        pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once each Agent execution completes, all calls in your LangChain object will be logged to Weights & Biases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### LangChain Context Manager
    Depending on your use case, you might instead prefer to use a context manager to manage your logging to W&B:
    """)
    return


@app.cell
def _(math_agent, os):
    from langchain.callbacks import wandb_tracing_enabled

    # unset the environment variable and use a context manager instead
    if "LANGCHAIN_WANDB_TRACING" in os.environ:
        del os.environ["LANGCHAIN_WANDB_TRACING"]

    # enable tracing using a context manager
    with wandb_tracing_enabled():
        math_agent.run("What is 5 raised to .123243 power?")  # this should be traced

    math_agent.run("What is 2 raised to .123243 power?")  # this should not be traced
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Non-Lang Chain Implementation

    A W&B Trace is created by logging 1 or more "spans". A root span is expected, which can accept nested child spans, which can in turn accept their own child spans. A Span represents a unit of work, Spans can have type `AGENT`, `TOOL`, `LLM` or `CHAIN`

    When logging with Trace, a single W&B run can have multiple calls to a LLM, Tool, Chain or Agent logged to it, there is no need to start a new W&B run after each generation from your model or pipeline, instead each call will be appended to the Trace Table.

    In this quickstart, we will how to log a single call to an OpenAI model to W&B Trace as a single span. Then we will show how to log a more complex series of nested spans.

    ## Logging with W&B Trace
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Call wandb.init to start a W&B run. Here you can pass a W&B project name as well as an entity name (if logging to a W&B Team), as well as a config and more. See wandb.init for the full list of arguments.

    You will see a Weights & Biases run start and be prompted to create a new API key at [wandb.ai/settings](https://wandb.ai/settings) if you haven't already. Store your API key securely. It can only be viewed once when created. Once you enter your API key, the inputs and outputs of your Agent calls will start to be streamed to the Weights & Biases App.

    **Note:** A W&B run supports logging as many traces you needed to a single run, i.e. you can make multiple calls of `run.log` without the need to create a new run each time
    """)
    return


@app.cell
def _():
    import wandb

    # start a wandb run to log to
    wandb.init(project="trace-example")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also set the entity argument in wandb.init if logging to a W&B Team.

    ### Logging a single Span
    Now we will query OpenAI times and log the results to a W&B Trace. We will log the inputs and outputs, start and end times, whether the OpenAI call was successful, the token usage, and additional metadata.

    You can see the full description of the arguments to the Trace class [here](https://soumik12345.github.io/wandb-addons/prompts/tracer/).
    """)
    return


@app.cell
def _(os):
    import openai
    import datetime
    from wandb.sdk.data_types.trace_tree import Trace
    openai.api_key = os.environ['OPENAI_API_KEY']
    model_name = 'gpt-3.5-turbo'
    temperature = 0.7
    # define your conifg
    system_message = 'You are a helpful assistant that always replies in 3 concise bullet points using markdown.'
    queries_ls = ['What is the capital of France?', 'How do I boil an egg?' * 10000, 'What to do if the aliens arrive?']
    for _query in queries_ls:
        _messages = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': _query}]
        _start_time_ms = datetime.datetime.now().timestamp() * 1000
        try:
            _response = openai.ChatCompletion.create(model=model_name, messages=_messages, temperature=temperature)  # deliberately trigger an openai error
            end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
            status = 'success'
            status_message = (None,)
            _response_text = _response['choices'][0]['message']['content']
            _token_usage = _response['usage'].to_dict()
        except Exception as e:
            end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
            status = 'error'
            status_message = str(e)
            _response_text = ''
            _token_usage = {}
        _root_span = Trace(name='root_span', kind='llm', status_code=status, status_message=status_message, metadata={'temperature': temperature, 'token_usage': _token_usage, 'model_name': model_name}, start_time_ms=_start_time_ms, end_time_ms=end_time_ms, inputs={'system_prompt': system_message, 'query': _query}, outputs={'response': _response_text})
        _root_span.log(name='openai_trace')  # logged in milliseconds  # create a span in wandb  # kind can be "llm", "chain", "agent" or "tool"  # log the span to wandb
    return Trace, datetime, model_name, openai, system_message, temperature


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logging a LLM pipeline using nested Spans

    In this example we will simulate an Agent being called, which then calls a LLM Chain, which calls an OpenAI LLM and then the Agent "calls" a Calculator tool.

    The inputs, outputs and metadata for each step in the execution of our "Agent" is logged in its own span. Spans can have child
    """)
    return


@app.cell
def _(Trace, datetime, model_name, openai, os, system_message, temperature):
    import time
    openai.api_key = os.environ['OPENAI_API_KEY']
    _query = 'How many days until the next US election?'
    _start_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    # The query our agent has to answer
    _root_span = Trace(name='MyAgent', kind='agent', start_time_ms=_start_time_ms, metadata={'user': 'optimus_12'})
    chain_span = Trace(name='LLMChain', kind='chain', start_time_ms=_start_time_ms)
    # part 1 - an Agent is started...
    _root_span.add_child(chain_span)
    _messages = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': _query}]
    _response = openai.ChatCompletion.create(model=model_name, messages=_messages, temperature=temperature)
    llm_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    _response_text = _response['choices'][0]['message']['content']
    _token_usage = _response['usage'].to_dict()
    llm_span = Trace(name='OpenAI', kind='llm', status_code='success', metadata={'temperature': temperature, 'token_usage': _token_usage, 'model_name': model_name}, start_time_ms=_start_time_ms, end_time_ms=llm_end_time_ms, inputs={'system_prompt': system_message, 'query': _query}, outputs={'response': _response_text})
    chain_span.add_child(llm_span)
    chain_span.add_inputs_and_outputs(inputs={'query': _query}, outputs={'response': _response_text})
    # part 2 - The Agent calls into a LLMChain..
    chain_span._span.end_time_ms = llm_end_time_ms
    time.sleep(3)
    days_to_election = 117
    tool_end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
    tool_span = Trace(name='Calculator', kind='tool', status_code='success', start_time_ms=llm_end_time_ms, end_time_ms=tool_end_time_ms, inputs={'input': _response_text}, outputs={'result': days_to_election})
    # add the Chain span as a child of the root
    _root_span.add_child(tool_span)
    _root_span.add_inputs_and_outputs(inputs={'query': _query}, outputs={'result': days_to_election})
    _root_span._span.end_time_ms = tool_end_time_ms
    # part 3 - the LLMChain calls an OpenAI LLM...
    # add the LLM span as a child of the Chain span...
    # update the end time of the Chain span
    # update the Chain span's end time
    # part 4 - the Agent then calls a Tool...
    # create a Tool span 
    # add the TOOL span as a child of the root
    # part 5 - the final results from the tool are added 
    # part 6 - log all spans to W&B by logging the root span
    _root_span.log(name='openai_trace')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once each Agent execution completes, all calls in your LangChain object will be logged to Weights & Biases
    """)
    return


if __name__ == "__main__":
    app.run()
