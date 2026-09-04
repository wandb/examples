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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/azure/azure-weave-cookbook-colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{azure-weave-cookbook-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install weave openai
    return


@app.cell
def _():
    model_id = "gpt-4-turbo" # @param {type:"string"}
    model_id = "mistral-7b-instruct-weave"
    azure_model_option = "openai" # @param ["openai", "ai_studio"]
    return azure_model_option, model_id


@app.cell
def _():
    wandb_entity = "a-sh0ts" # @param {type:"string"}
    weave_project = "azure-weave-cookbook" # @param {type:"string"}
    eval_dataset_name = "customer_service_inquiries" # @param {type:"string"}
    publish_eval_data = True # @param {type:"boolean"}
    return eval_dataset_name, publish_eval_data, wandb_entity, weave_project


@app.cell
def _(azure_model_option):
    from google.colab import userdata
    import os
    from openai import AzureOpenAI, OpenAI

    os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY')

    if azure_model_option == "openai":
        os.environ["AZURE_OPENAI_ENDPOINT"] = userdata.get('AZURE_OPENAI_ENDPOINT')
        os.environ["AZURE_OPENAI_API_KEY"] = userdata.get('AZURE_OPENAI_API_KEY')
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    elif azure_model_option == "ai_studio":
        os.environ["AZURE_AI_STUDIO_API_ENDPOINT"] = userdata.get('AZURE_AI_STUDIO_API_ENDPOINT')
        os.environ["AZURE_AI_STUDIO_API_KEY"] = userdata.get('AZURE_AI_STUDIO_API_KEY')

        api_version = "v1"
        client = OpenAI(
            base_url=f"{os.getenv('AZURE_AI_STUDIO_API_ENDPOINT')}/v1",
            api_key=os.getenv('AZURE_AI_STUDIO_API_KEY')
        )
    else:
        print("Please us one of the above options")
    return (client,)


@app.cell
def _(weave_project):
    import weave
    weave.init(weave_project)
    return (weave,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calling Azure directly
    """)
    return


@app.cell
def _(client, weave):
    @weave.op()
    def call_azure_chat(model_id: str, messages: list, max_tokens: int = 1000, temperature: float = 0.5):
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return {"status": "success", "response": response.choices[0].message.content}

    return (call_azure_chat,)


@app.cell
def _(weave):
    @weave.op()
    def format_messages_for_mistral(messages: list):
        system_message = messages[0]["content"]
        formatted_messages = []

        for message in messages[1:]:
            if message["role"] == "user":
                formatted_message = {
                    "role": "user",
                    "content": f"[INST]\n{system_message}\n{message['content']}\n[/INST]"
                }
            else:
                formatted_message = {
                    "role": message["role"],
                    "content": message["content"]
                }
            formatted_messages.append(formatted_message)

        return formatted_messages

    return (format_messages_for_mistral,)


@app.cell
def _(call_azure_chat, format_messages_for_mistral, model_id):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Create a snack recipe for a dish called the Azure Weav-e-ohs"}
    ]
    if "mistral" in model_id.lower():
        messages = format_messages_for_mistral(messages)
    result = call_azure_chat(model_id, messages)
    print(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating Functional LLM Apps
    """)
    return


@app.cell
def _(weave):
    @weave.op()
    def format_prompt(prompt: str):
        "A formatting function for OpenAI models"
        system_prompt_formatted = "You are a helpful assistant."

        human_prompt = """
        {prompt}
        """

        human_prompt_formatted = human_prompt.format(prompt=prompt)
        messages = [{"role":"system", "content":system_prompt_formatted}, {"role":"user", "content":human_prompt_formatted}]
        return messages

    return (format_prompt,)


@app.cell
def _(
    call_azure_chat,
    format_messages_for_mistral,
    format_prompt,
    model,
    weave,
):
    @weave.op()
    def run_chat(model_id: str, prompt: str):
        formatted_messages = format_prompt(prompt=prompt)
        if "mistral" in model.lower():
            formatted_messages = format_messages_for_mistral(formatted_messages)
        result = call_azure_chat(model_id, formatted_messages, max_tokens=1000)
        return result

    return (run_chat,)


@app.cell
def _():
    prompt = "Give a full recipe for a Weights & Biases inspired cocktail. Ensure you provide a list of ingredients, tools, and step by step instructions"
    return (prompt,)


@app.cell
def _(model_id, prompt, run_chat):
    result_1 = run_chat(model_id, prompt)
    return (result_1,)


@app.cell
def _(result_1):
    result_1['response']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create LLM Model Classes to iterate over hyperparameters
    """)
    return


@app.cell
def _(weave):
    from dataclasses import dataclass

    @dataclass
    class PromptTemplate:
        system_prompt: str
        human_prompt: str

        @weave.op()
        def format_prompt(self, email_content: str):
            "A formatting function for OpenAI models"
            system_prompt_formatted = self.system_prompt.format()
            human_prompt_formatted = self.human_prompt.format(email_content=email_content)
            messages = [{"role":"system", "content":system_prompt_formatted}, {"role":"user", "content":human_prompt_formatted}]
            return messages

    return (PromptTemplate,)


@app.cell
def _(
    PromptTemplate,
    call_azure_chat,
    format_messages_for_mistral,
    model_id,
    weave,
):
    from weave import Model
    from typing import Tuple

    class AzureEmailAssistant(Model):
        model_id: str = model_id
        prompt_template: PromptTemplate
        max_tokens: int = 2048
        temperature: float = 0.0

        @weave.op()
        def format_doc(self, doc: str) -> list:
            "Read and format the document"
            messages = self.prompt_template.format_prompt(doc)
            return messages

        @weave.op()
        def respond(self, doc: str) -> dict:
            "Generate a response to the email inquiry"
            messages = self.format_doc(doc)
            if "mistral" in self.model_id.lower():
                messages = format_messages_for_mistral(messages)
            output = call_azure_chat(
                self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature)
            return output

        @weave.op()
        async def predict(self, email_content: str) -> str:
            return self.respond(email_content)["response"]

    return (AzureEmailAssistant,)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%writefile customer_inquiry.txt
    # Subject: Inquiry about Order Delay
    # 
    # Hello,
    # 
    # I placed an order last week for the new UltraGlow Skin Serum, but I have not received a shipping update yet. My order number is 12345. Could you please update me on the status of my shipment?
    # 
    # Thank you,
    # Jane Doe
    return


@app.cell
def _():
    system_prompt = """
    # Instructions
    You are a customer service response assistant. Our goal is to provide clear, concise, and polite responses to customer inquiries about products, shipping, and any issues they may have encountered. Some rules to remember:
    - Always be courteous and respectful.
    - Provide accurate and helpful information.
    - Responses should be concise and to the point.
    - Use formal language suitable for professional communication.
    ## Formatting Rules
    Maintain a formal greeting and closing in each response. Do not use slang or overly casual language. Ensure all provided information is correct and double-check for typographical errors.
    """

    human_prompt = """
    Here is a customer inquiry received via email. Craft a suitable response based on the guidelines provided:
    <Customer Inquiry>
    {email_content}
    <End of Inquiry>
    """
    return human_prompt, system_prompt


@app.cell
def _(PromptTemplate, human_prompt, system_prompt):
    prompt_template = PromptTemplate(
        system_prompt=system_prompt,
        human_prompt=human_prompt)
    return (prompt_template,)


@app.cell
def _(weave, weave_project):
    weave.init(weave_project) # Colab specific
    return


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(AzureEmailAssistant, Path, model_id, prompt_template):
    doc = Path('customer_inquiry.txt').read_text()
    model = AzureEmailAssistant(model_id=model_id, prompt_template=prompt_template)
    response = model.respond(doc)
    return model, response


@app.cell
def _(response):
    print(response["response"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Optional] Publish synthetically generated Evaluation data to Weave
    """)
    return


@app.cell
def _(eval_dataset_name, publish_eval_data, weave):
    if publish_eval_data:
        from weave import Dataset
        dataset = Dataset(name=eval_dataset_name, rows=[
            {'id': '1', 'email_content': 'Subject: Inquiry about Order Delay\n\nHello,\n\nI placed an order last week for the new UltraGlow Skin Serum, but I have not received a shipping update yet. My order number is 12345. Could you please update me on the status of my shipment?\n\nThank you,\nJane Doe'},
            {'id': '2', 'email_content': 'Subject: Damaged Item Received\n\nHello,\n\nI received my order yesterday, but one of the items, a glass vase, was broken. My order number is 67890. How can I get a replacement or a refund?\n\nBest regards,\nJohn Smith'},
            {'id': '3', 'email_content': 'Subject: Wrong Item Delivered\n\nHi,\n\nI ordered a pair of blue sneakers, but I received a black pair instead. My order number is 54321. Could you please assist me with this issue?\n\nThank you,\nEmily Johnson'},
            {'id': '4', 'email_content': 'Subject: Request for Return Instructions\n\nDear Customer Service,\n\nI would like to return a dress I purchased last week as it does not fit well. My order number is 11223. Could you please provide the return instructions?\n\nSincerely,\nLaura Davis'},
            {'id': '5', 'email_content': 'Subject: Missing Items in Order\n\nHello,\n\nI just received my order, but two items are missing. My order number is 33445. Could you please help me resolve this?\n\nKind regards,\nMichael Brown'},
            {'id': '6', 'email_content': 'Subject: Delay in Order Confirmation\n\nDear Support Team,\n\nI placed an order two days ago but have not received a confirmation email yet. My order number is 99887. Can you confirm if my order was processed?\n\nThank you,\nSarah Wilson'},
            {'id': '7', 'email_content': 'Subject: Inquiry About Product Availability\n\nHi,\n\nI\'m interested in purchasing the Professional Chef Knife Set, but it appears to be out of stock. Can you let me know when it will be available again?\n\nBest regards,\nDavid Martinez'},
            {'id': '8', 'email_content': 'Subject: Request for Invoice\n\nDear Customer Service,\n\nCould you please send me an invoice for my recent purchase? My order number is 55667. I need it for my records.\n\nThank you,\nJessica Taylor'},
            {'id': '9', 'email_content': 'Subject: Issue with Discount Code\n\nHello,\n\nI tried using the discount code SAVE20 during checkout, but it did not apply. My order number is 77654. Could you please assist me?\n\nSincerely,\nRobert Anderson'},
            {'id': '10', 'email_content': 'Subject: Request for Expedited Shipping\n\nHi,\n\nI need my order delivered urgently. Is it possible to upgrade to expedited shipping? My order number is 44556.\n\nThank you,\nLinda Thompson'},
            {'id': '11', 'email_content': 'Subject: Order Cancellation Request\n\nDear Support Team,\n\nI would like to cancel my recent order as I made a mistake while ordering. My order number is 33221. Can you please process the cancellation?\n\nBest regards,\nWilliam Clark'}
        ])
        # Publish the dataset
        weave.publish(dataset)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Evaluation whilst logging results to Weave
    """)
    return


@app.cell
def _(eval_dataset_name, wandb_entity, weave, weave_project):
    dataset_uri = f'weave:///{wandb_entity}/{weave_project}/object/{eval_dataset_name}:latest'
    dataset_1 = weave.ref(dataset_uri).get()
    return (dataset_1,)


@app.cell
def _(weave):
    # Scoring function checking length of summary
    @weave.op()
    def check_conciseness(model_output: str) -> dict:
        result = len(model_output.split()) < 300
        return {'conciseness': result}

    return (check_conciseness,)


@app.cell
def _(check_conciseness, dataset_1, weave):
    evaluation = weave.Evaluation(dataset=dataset_1, scorers=[check_conciseness])
    return (evaluation,)


@app.cell
async def _(evaluation, model):
    await evaluation.evaluate(model)
    return


if __name__ == "__main__":
    app.run()
