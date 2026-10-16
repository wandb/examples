# Arxiv PDF Summarization Bot using Chain of Density

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/weave/blob/add-summarization-example/examples/cookbooks/summarization/chain_of_density_arxiv.ipynb)
![Eval Comparison](./media/eval_comparison.gif)

This cookbook walks through the implementation of an AI-powered summarization bot that extracts concise, information-dense summaries from Arxiv papers using the Chain of Density technique. We'll use Anthropic's Claude API, the Arxiv API, PyPDF2 for PDF processing, and Weave for experiment tracking and evaluation.

## Setup and Imports

First, let's set up our environment and import the necessary libraries we'll need for our project, including:
- `anthropic` for interacting with Claude API
- `arxiv` for fetching paper metadata and PDFs
- `PyPDF2` for PDF text extraction
- `weave` for experiment tracking and evaluation

## Initializing Weave and Anthropic Client

Next, we initialize Weave for experiment tracking and set up the Anthropic client:

```python
weave.init("arxiv-chain-of-density-summarization")
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

This sets up Weave with a specific project name and initializes the Anthropic client using an API key stored in the environment variables.

## Optional: Fetching Arxiv Papers

<details>
<summary>Click to Expand</summary>

We implement functions to fetch relevant papers from the Arxiv database:

![Generate Arxiv Query](./media/generate_arxiv_query_args.gif)

```python
@weave.op()
def generate_arxiv_query_args(instruction, model="claude-3-sonnet-20240229"):
    # Define the tools available to the LLM
    tools = [{
        "name": "prepare_arxiv_search",
        "description": "Prepare arguments for ArXiv paper search. This tool generates an optimal query string utilizing Boolean operators, field-specific syntax, and precise search terms. It also determines an efficient maximum number of results to fetch, balancing comprehensive coverage with processing efficiency. The output is tailored to the given research instruction, aiming to provide relevant and focused search results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The ArXiv search query string. Supports Boolean operators (AND, OR, NOT), field-specific syntax (e.g., 'ti:' for title, 'au:' for author), quotation marks for exact phrases, and wildcards. Can include multiple search terms to refine results based on title, abstract, authors, comments, journal reference, subject category, or report number."
                },
                "max_results": {
                    "type": "integer",
                    "description": "The maximum number of paper results to return from the ArXiv search. Aims to minimize the number of results while ensuring sufficient coverage of the topic. Defaults to 5 if not specified. Increasing this value broadens the search but may increase processing time and resource usage. Aim to be below 10 articles."
                }
            },
            "required": ["query", "max_results"]
        }
    }]

    # Define the system prompt for the LLM
    system_prompt = """You are an expert at generating ArXiv queries. Use the prepare_arxiv_search tool to create an optimal query and determine the appropriate maximum number of results for the given research question. The query should utilize advanced search techniques including Boolean operators, field-specific syntax, and precise terms to ensure comprehensive yet focused results."""

    # Create the user message with the instruction
    messages = [
        {
            "role": "user",
            "content": f"Use the prepare_arxiv_search tool to generate an optimal ArXiv query and determine the maximum number of results for the following research instruction: {instruction}"
        }
    ]

    # Make the API call to the LLM
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=messages,
        system=system_prompt,
        tools=tools
    )

    # Extract the query and max_results from the response
    for content in response.content:
        if content.type == 'tool_use' and content.name == 'prepare_arxiv_search':
            args = content.input
            return args.get('query'), args.get('max_results')

    # If no tool use was found, return a default query and the provided max_results
    return f"{instruction}", 5
```

![Fetch Arxiv Papers](./media/fetch_arxiv_papers.gif)

```python
@weave.op()
def fetch_arxiv_papers(query, max_results=5):
    # Initialize the arxiv Client
    arxiv_client = arxiv.Client()
    
    # Create the search object with the provided query and max_results
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # Fetch the results using client.results() and convert them to ArxivPaper objects
    papers = []
    for result in arxiv_client.results(search):
        # Convert the raw arxiv result to our custom ArxivPaper object
        paper = convert_raw_arxiv_to_pydantic(result)
        papers.append(paper)
    
    return papers
```

These functions use Claude to generate an optimal Arxiv search query based on a given instruction and then fetch the relevant papers using the Arxiv API.
</details>


## Creating Sample Arxiv Paper Objects

For demonstration purposes, we create sample `ArxivPaper` objects:

![Create Sample Arxiv Paper Objects](./media/arxiv_paper.gif)

```python
arxiv_paper = ArxivPaper(
    entry_id="http://arxiv.org/abs/2406.04744v1",
    updated=datetime(2024, 6, 7, 8, 43, 7, tzinfo=timezone.utc),
    published=datetime(2024, 6, 7, 8, 43, 7, tzinfo=timezone.utc),
    title="CRAG -- Comprehensive RAG Benchmark",
    authors=[Author(full_name="Xiao Yang")],
    summary="CRAG: A benchmark for Retrieval-Augmented Generation (RAG) with 4,409 QA pairs across diverse domains.",
    doi="10.48550/arXiv.2406.04744",
    primary_category="cs.CL",
    pdf_url="https://arxiv.org/pdf/2406.04744"
)
```

This creates an `ArxivPaper` object with metadata about a specific paper, including its title, authors, summary, and PDF URL. The most important part of this object is the `pdf_url` field, which contains the location of the PDF file.

## PDF Processing

We implement functions to load and process PDFs:

```python
def load_pdf(arxiv_result):
    pdf_url = arxiv_result["pdf_url"]
    response = requests.get(pdf_url)
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return pdf_reader
```

These functions handle loading PDFs into PyPDF2 making processing the PDFs easier and more efficient.

## Converting PDF Images to Text

One of the key challenges in processing academic papers is handling the visual content, which often includes both raster images and vector graphics. These visuals can contain crucial information that needs to be incorporated into our summarization process. To address this, we leverage Claude 3 Sonnet's advanced vision capabilities to convert these images into detailed textual descriptions.

Here's the implementation of our main image processing function:

![Extract Images](./media/model_extract_images.gif)

```python
import base64
import io
from pdf2image import convert_from_bytes
from PIL import Image

@weave.op()
def extract_images(paper, model="claude-3-5-sonnet-20240620"):
    pdf_reader = load_pdf(paper)
    all_images = []

    for page_num, page in enumerate(pdf_reader.pages):
        images = []

        # Process raster images
        for image in page.images:
            img_data = image.data
            kind = filetype.guess(img_data)
            if kind is None:
                print(f"Cannot guess file type for image on page {page_num + 1}")
                continue
            
            img_str = base64.b64encode(img_data).decode("utf-8")
            data_url = f"data:{kind.mime};base64,{img_str}"
            try:
                images.append(
                    {"image": data_url, "description": process_figure_image(data_url, model=model)}
                )
            except Exception as e:
                print(f"Error processing image on page {page_num + 1}: {e}")
                images.append({"image": data_url, "description": ""})
        
        # Process vector graphics
        vector_graphics_image_data_url = convert_vector_graphic_page_to_image(page)
        if vector_graphics_image_data_url:
            images.append({
                "image": vector_graphics_image_data_url, 
                "description": process_vector_image_pdf(vector_graphics_image_data_url, model=model)
            })
        
        all_images.append(images)

    return all_images
```

Let's break down the key components and challenges:

### 1. Handling Raster Images

Raster images are typically stored as embedded objects within the PDF. We extract these using PyPDF2's built-in functionality:

```python
for image in page.images:
    img_data = image.data
    # ... process the image data
```

The challenge here is that these images can be in various formats (PNG, JPEG, etc.). We use the `filetype` library to guess the MIME type, which is crucial for creating a valid data URL:

```python
kind = filetype.guess(img_data)
if kind is None:
    print(f"Cannot guess file type for image on page {page_num + 1}")
    continue

img_str = base64.b64encode(img_data).decode("utf-8")
data_url = f"data:{kind.mime};base64,{img_str}"
```

### 2. Handling Vector Graphics

Vector graphics present a unique challenge because they're not stored as traditional image files within the PDF. Instead, they're often part of the page's content stream. To handle these, we need to convert the entire page to an image:

```python
vector_graphics_image_data_url = convert_vector_graphic_page_to_image(page)
if vector_graphics_image_data_url:
    images.append({
        "image": vector_graphics_image_data_url, 
        "description": process_vector_image_pdf(vector_graphics_image_data_url, model=model)
    })
```

The `convert_vector_graphic_page_to_image` function (collapsed below) uses `pdf2image` to convert the PDF page to a PNG image. This ensures we capture all vector graphics, but it also means we might capture text and other elements on the page.

<details>
<summary>Click to Expand</summary>

```python
def convert_vector_graphic_page_to_image(pdf_page, scale_factor=0.5):
    # Helper function to handle indirect PDF objects
    def get_object(obj):
        if isinstance(obj, PyPDF2.generic.IndirectObject):
            return obj.get_object()
        return obj

    # Extract resources from the PDF page
    resources = get_object(pdf_page.get('/Resources', {}))
    xobject = get_object(resources.get('/XObject', {}))
    
    # Check if there's a figure that's not a raster image (i.e., a vector graphic)
    if xobject:
        for obj in xobject.values():
            obj = get_object(obj)
            # Check if the object is a Form XObject, which typically represents vector graphics
            if isinstance(obj, dict) and obj.get('/Subtype') == '/Form':
                # Convert the page to a temporary PDF file in memory
                pdf_bytes = io.BytesIO()
                pdf_writer = PyPDF2.PdfWriter()
                pdf_writer.add_page(pdf_page)
                pdf_writer.write(pdf_bytes)
                pdf_bytes.seek(0)
                
                # Use pdf2image to convert the PDF to a PNG image
                images = convert_from_bytes(pdf_bytes.getvalue(), fmt='png')
                
                if images:
                    image = images[0]
                    # Resize the image to reduce memory usage and processing time
                    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                    image = image.resize(new_size, Image.LANCZOS)
                    
                    # Convert the image to a base64-encoded string
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    img_str = base64.b64encode(img_byte_arr).decode("utf-8")
                    
                    # Return the image as a data URL
                    return f"data:image/png;base64,{img_str}"
    
    # Return None if no vector graphics were found or conversion was not needed
    return None
```

This approach ensures that all vector graphics on the page are captured, even if they can't be directly extracted as separate objects. However, it's important to note that this method will also capture all other content on the page, which may require additional processing or filtering in subsequent steps of the analysis pipeline.

</details>


### 3. Using Claude 3 Sonnet for Image Description

The core of our image processing lies in the `process_figure_image` and `process_vector_image_pdf` functions. These functions use Claude 3 Sonnet's vision capabilities to generate detailed descriptions of the images:

```python
@weave.op()
def process_figure_image(data_url, model="claude-3-5-sonnet-20240620"):
    img_str = data_url.split(",")[1]

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_str,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image as if it's a figure from a scientific research paper. Provide a detailed technical description addressing the following:

1. Type of figure (e.g., graph, diagram, flowchart, experimental setup)
2. Key components or variables represented
3. Relationships or trends depicted
4. Quantitative information (if present)
5. Methodology or process illustrated (if applicable)
6. Potential implications or conclusions that can be drawn
7. Any limitations or assumptions evident in the figure

Focus on technical accuracy and relevance to scientific research. Avoid general descriptions and concentrate on the specific scientific content presented.""",
                    },
                ],
            }
        ],
    )
    return response.content[0].text

@weave.op()
def process_vector_image_pdf(data_url, model="claude-3-5-sonnet-20240620"):
    img_str = data_url.split(",")[1]

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_str,
                        },
                    },
                    {
                        "type": "text",
                        "text": """This image is a full page from a scientific paper PDF, converted to PNG format. It may contain one or more vector graphic figures or charts. Your task is to:

1. Identify and focus solely on the vector graphic figures or charts within the page.
2. For each identified figure or chart, provide a detailed technical analysis addressing:

   a. Type of figure (e.g., graph, diagram, flowchart)
   b. Key components or variables represented
   c. Relationships or trends depicted
   d. Quantitative information (if present)
   e. Methodology or process illustrated (if applicable)
   f. Potential implications or conclusions that can be drawn

3. Ignore any text or other elements on the page that are not part of the vector graphic figures.
4. If multiple figures are present, analyze each separately and clearly indicate which figure you are describing.

Focus on providing accurate, technical descriptions of the vector graphic content only.""",
                    },
                ],
            }
        ],
    )
    return response.content[0].text
```

> The prompts for `process_figure_image` and `process_vector_image_pdf` are tailored to handle different scenarios:
> 
> 1. **Figure Image Prompt:**
>    - Assumes a single, isolated figure
>    - Focuses on detailed analysis of the specific figure
>    - Includes points about limitations and assumptions
> 
> 2. **Vector Image PDF Prompt:**
>    - Assumes a full page that may contain multiple vector graphics
>    - Instructs to identify and focus only on vector graphic elements
>    - Asks for separate analysis of each figure if multiple are present
>    - Explicitly tells to ignore text and non-vector graphic elements
> 
> These differences ensure that Claude 3 Sonnet can accurately process and describe both individual figures and complex pages with multiple vector graphics.

This approach allows us to handle the nuances of different image types within scientific papers. The figure image prompt is designed for standalone images, while the vector image prompt is tailored for full pages that may contain multiple graphics alongside text and other elements.

### 4. Integrating Image Descriptions into the Text

Finally, we integrate the image descriptions into the text of the paper:

![Replace Images with Descriptions](./media/model_replace_image_with_descriptions.gif)

```python
@weave.op()
def replace_images_with_descriptions(paper, images):
    # ... (previous code)
    if images[page_num] and len(images[page_num]) > 0:
        text += f"\n\n[Image Descriptions for page {page_num+1}]\n"
        for image_num, image in enumerate(images[page_num]):
            text += f"\n[Image {image_num+1}]: {image['description']}\n"
        text += "[END OF IMAGE DESCRIPTIONS]\n"
    # ... (rest of the function)
```

This approach ensures that the image descriptions are clearly demarcated within the text, making it easier for our summarization pipeline to incorporate this visual information.

By implementing this comprehensive image processing pipeline, we ensure that our Chain of Density summarization process can incorporate crucial information from both textual and visual elements of academic papers. This is particularly important for fields where figures and diagrams play a significant role in conveying research findings.

## Chain of Density Summarization

The core of our summarization pipeline is then implemented in the following functions:

![Chain of Density Summarization](./media/chain_of_density.gif)

### `summarize_current_summary`:
  - Forms the foundation of our Chain of Density implementation
  - Utilizes a carefully crafted prompt to guide the language model
  - Instructs the model to identify new technical entities
  - Incorporates new entities into the summary
  - Increases overall information density while maintaining relevance to the given instruction

```python
@weave.op()
def summarize_current_summary(document, instruction, current_summary="", iteration=1, model="claude-3-5-sonnet-20240620"):
    # Define the maximum number of tokens for the model's response
    max_tokens = 4096

    # Construct the prompt for the LLM
    prompt = f"""
    Document:
    {document}
    
    Current summary:
    {current_summary}

    Instruction to focus on: {instruction}

    Iteration: {iteration}

    Generate an increasingly concise, entity-dense, and highly technical summary from the provided document that specifically addresses the given instruction using the below approach:

    1. Carefully read the current summary and the instruction.

    2. Identify 1-3 new, important technical entities or ideas from the original text that:
       - Are directly relevant to the instruction
       - Are not yet present in the current summary
       - Add significant, specific information to the summary
       - Are preferably 5 words or fewer
       - May include methodologies, algorithms, metrics, or key findings
       - Ensure to include this in the output before the summary

    3. Write a new summary that:
       - Incorporates the newly identified entities/ideas
       - Retains all crucial information from the current summary
       - Increases overall information density
       - Remains focused on addressing the instruction
       - Utilizes the response window of {max_tokens} tokens

    Guidelines:
    - Prioritize technical accuracy and specificity over general readability
    - Use precise terminology, domain-specific jargon, and include quantitative details where relevant
    - Ensure all information is directly related to the instruction
    - Make every word count: rewrite to improve density and make space for new technical entities
    - Employ fusion, compression, and removal of less informative phrases to increase density
    - Never drop entities or technical details from the current summary that are relevant to the instruction
    - Maintain coherence while maximizing information density

    Your goal is to create a summary that is noticeably denser, more technical, and more informative than the previous one, utilizing the response window of {max_tokens} tokens while staying laser-focused on the instruction. The summary should be suitable for an expert audience in the field."""

    # Make the API call to the LLM
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Return the generated summary
    return response.content[0].text
```

### `iterative_density_summarization`:
  - Orchestrates the iterative refinement process
  - Repeatedly calls `summarize_current_summary`
  - Uses each iteration's output as input for the next
  - Allows for gradual accumulation of technical details
  - Increases density of information progressively

```python
@weave.op()
def iterative_density_summarization(document, instruction, current_summary, density_iterations, model):
    # Initialize a list to store summaries from each iteration
    iteration_summaries = []
    
    # Iterate through the specified number of density iterations
    for iteration in range(1, density_iterations + 1):
        # Generate a new summary based on the current summary and document
        current_summary = summarize_current_summary(document, instruction, current_summary, iteration, model)
        
        # Add the new summary to the list of iteration summaries
        iteration_summaries.append(current_summary)
        
        # Print the current iteration and summary for monitoring
        print(f"Iteration {iteration}:\n{current_summary}\n")
    
    # Return the final summary and the list of all iteration summaries
    return current_summary, iteration_summaries
```

### `final_summary`:
  - Performs a final condensation step after the iterative process
  - Aims to reduce summary length by 30-40%
  - Retains all critical technical content
  - Optimizes for maximum information density and relevance to the instruction

```python
@weave.op()
def final_summary(instruction, current_summary, model):
    # Construct the prompt for the final summary generation
    prompt = f"""Given this summary:

{current_summary}

And this instruction to focus on:

{instruction}

Create an extremely dense, final summary that captures all key technical information in the most concise form possible, while specifically addressing the given instruction. Follow these guidelines:

1. Aim to reduce length by 30-40% while retaining all critical technical content relevant to the instruction.
2. Prioritize highly specific methodologies, algorithms, metrics, and findings that directly address the instruction.
3. Preserve precise quantitative data, including statistical significance and error margins where applicable and relevant to the instruction.
4. Maintain the use of domain-specific terminology and technical jargon pertinent to the instruction.
5. Ensure that all key entities and concepts from the original summary that relate to the instruction are represented.
6. Use compact phrasing and remove any remaining non-essential information that doesn't directly contribute to addressing the instruction.
7. If relevant to the instruction, include brief mentions of limitations, assumptions, or conflicting viewpoints.
8. Optimize for information density while maintaining coherence for an expert audience, always keeping the focus on the given instruction.

The final summary should be a highly concentrated, technical distillation of the research that specifically addresses the given instruction, suitable for specialists in the field."""

    # Make the API call to the LLM for the final summary
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Return the generated final summary
    return response.content[0].text
```

### `chain_of_density_summarization`:
  - Serves as the main entry point for the summarization process
  - Coordinates the entire summarization pipeline
  - Initiates the iterative summarization
  - Applies the final condensation
  - Returns a comprehensive result set including:
    - Final summary
    - Accumulated summary
    - All intermediate summaries

```python
@weave.op()
def chain_of_density_summarization(document, instruction, current_summary="", model="claude-3-5-sonnet-20240620", density_iterations=2):
    # Perform iterative density summarization
    current_summary, iteration_summaries = iterative_density_summarization(document, instruction, current_summary, density_iterations, model)
    
    # Generate the final, highly condensed summary
    final_summary_text = final_summary(instruction, current_summary, model)
    
    # Print the final summary for monitoring
    print(f"Final Summary:\n{final_summary_text}\n")

    # Return a dictionary containing all generated summaries
    return {
        "final_summary": final_summary_text,
        "accumulated_summary": current_summary,
        "iteration_summaries": iteration_summaries,
    }
```

This implementation leverages the Chain of Density technique to produce increasingly dense and informative summaries. By iteratively refining the summary and focusing on technical entities and ideas, it generates concise yet highly informative summaries tailored to specific instructions. The process prioritizes technical accuracy, domain-specific terminology, and quantitative details, making it particularly suitable for summarizing complex scientific documents for expert audiences.

## Weave Model Object

We create a Weave Model object to encapsulate our summarization pipeline:

![Weave Model Object](./media/model.gif)

```python
class ArxivChainOfDensityPipeline(weave.Model):
    model: str = "claude-3-5-sonnet-20240620"
    density_iterations: int = 3

    def __init__(self, model: str = "claude-3-5-sonnet-20240620", density_iterations: int = 3):
        super().__init__()
        self.model = model
        self.density_iterations = density_iterations

    @weave.op()
    def predict(self, paper: ArxivPaper, instruction: str) -> dict:
        extracted_images = extract_images(paper)
        cleaned_text = replace_images_with_descriptions(paper, extracted_images)
        result = chain_of_density_summarization(cleaned_text, instruction, model=self.model, density_iterations=self.density_iterations)
        return result
```

This class encapsulates our summarization pipeline as a Weave Model. By inheriting from `weave.Model` and using the `@weave.op()` decorator, we enable automatic versioning and tracking of inputs, outputs, and code changes. This makes it easy to reproduce experiments and compare results across different model versions or parameter settings.

## Evaluation Dataset

We create an evaluation dataset using sample Arxiv papers and instructions:

![Evaluation Dataset](./media/eval_dataset.gif)

```python
eval_papers = [arxiv_paper3]
eval_instructions = [
    "Summarize the key methodologies and novel contributions of this research, focusing on their potential impact in the field.",
]

eval_data = list(product(eval_papers, eval_instructions))
dataset = weave.Dataset(name="we-paper-reading-eval-data", rows=[{"paper": arxiv_paper, "instruction": instruction, "summary": arxiv_paper.summary} for arxiv_paper, instruction in eval_data])
weave.publish(dataset)
```

This creates a Weave Dataset object that combines papers, instructions, and original summaries for evaluation. The `weave.Dataset` class allows us to version and track our evaluation data, ensuring reproducibility of our experiments. By publishing the dataset with `weave.publish()`, we make it available for future use and comparison.

## Evaluation Metrics

We implement several evaluation metrics to assess the quality of our summaries:

![Evaluation Metrics](./media/evals_main_screen.gif)

```python
@weave.op()
def score_summary(summary, summary_type, instruction, model):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Construct a detailed prompt for the GPT model to evaluate the summary
    prompt = f"""Evaluate the quality of the following {summary_type} based on how well it addresses the given instruction. Use the scoring rules below to calculate three numerical scores between 0 and 10.

Instruction: {instruction}

{summary_type}:
{summary}

Scoring Rules:
1. Relevance (0-5): [Detailed scoring criteria for relevance]
2. Technical Quality (0-5): [Detailed scoring criteria for technical quality]
3. Conciseness (0-5): [Detailed scoring criteria for conciseness]

Provide your evaluation in the following JSON format:
{{
    "relevance": {{
        "score": <float>
    }},
    "technical_quality": {{
        "score": <float>
    }},
    "conciseness": {{
        "score": <float>
    }}
}}

Ensure your response is ONLY valid JSON. Do not include any other text outside the JSON object.
Ensure you have the keys: relevance, technical_quality, conciseness, each containing only a score.
Ensure each score is a float between 0 and 10, using the scoring rules provided above.
"""

    # Make an API call to the GPT model for evaluation
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    # Parse and return the JSON response
    return json.loads(response.choices[0].message.content)
```

This function uses GPT-4 to evaluate individual summaries based on three criteria:
- Relevance
- Technical quality
- Conciseness

Benefits:
- Captures nuanced aspects of summary quality
- Provides a holistic assessment of how well the summary addresses the given instruction
- Evaluates technical accuracy while considering conciseness

---

```python
@weave.op()
def calculate_long_tail_stats(scores):
    if not scores:
        return None
    
    aspects = ['relevance', 'technical_quality', 'conciseness']
    stats = {}
    
    for aspect in aspects:
        try:
            # Handle different input formats (list of lists or list of dicts)
            if isinstance(scores[0], list):
                flattened_scores = [score[aspect]['score'] for sublist in scores for score in sublist]
            elif isinstance(scores[0], dict):
                flattened_scores = [score[aspect]['score'] for score in scores]
            else:
                print(f"Unexpected format for scores: {scores}")
                return None
            
            # Calculate statistics for each aspect
            stats[aspect] = {
                "mean": np.mean(flattened_scores),
                "tail_ratio": np.mean(sorted(flattened_scores)[-max(1, int(len(flattened_scores)*0.05)):]) / np.mean(flattened_scores),
            }
        except Exception as e:
            print(f"Error calculating stats for {aspect}: {str(e)}")
            stats[aspect] = None
    
    return stats
```

This function:
- Analyzes the distribution of scores across multiple summaries
- Calculates for each aspect (relevance, technical quality, conciseness):
  - Mean score
  - "Tail ratio" (average of top 5% scores compared to overall mean)

Usefulness:
- Helps identify potential outliers or exceptionally high-quality summaries
- Provides insight into overall performance of the summarization process
- Highlights areas where the model excels or needs improvement

---

```python
@weave.op()
def analyze_iteration_impact(scores):
    if len(scores) < 2:
        return {aspect: {"diminishing_returns_point": 0, "cumulative_improvement": 0} for aspect in ['relevance', 'technical_quality', 'conciseness']}
    
    aspects = ['relevance', 'technical_quality', 'conciseness']
    results = {}
    
    for aspect in aspects:
        aspect_scores = [s[aspect]['score'] for s in scores]
        improvements = [aspect_scores[i+1] - aspect_scores[i] for i in range(len(aspect_scores)-1)]
        
        results[aspect] = {
            "diminishing_returns_point": next((i for i, imp in enumerate(improvements) if imp <= 0), len(improvements)),
            "cumulative_improvement": sum(improvements),
        }
    
    return results
```


This function:
- Assesses the improvement of summaries across iterations
- Key metrics:
  - Point of diminishing returns (where improvements become negative or zero)
  - Cumulative improvement for each aspect

Value:
- Helps optimize the number of iterations in the Chain of Density process
- Determines when further iterations may no longer yield significant improvements

---

```python
@weave.op()
def find_optimal_improvement_range(scores):
    if len(scores) < 3:
        return {aspect: {"optimal_range_start": 0, "optimal_range_end": 0, "score_at_start": 0, "score_at_end": 0, "improvement_in_range": 0} for aspect in ['relevance', 'technical_quality', 'conciseness']}
    
    aspects = ['relevance', 'technical_quality', 'conciseness']
    results = {}
    
    for aspect in aspects:
        aspect_scores = [s[aspect]['score'] for s in scores]
        improvements = [aspect_scores[i+1] - aspect_scores[i] for i in range(len(aspect_scores)-1)]
        
        # Calculate moving average of improvements
        window_size = min(3, len(aspect_scores) - 1)
        moving_avg = np.convolve(improvements, np.ones(window_size), 'valid') / window_size
        
        # Find range where improvements are above a threshold
        threshold = 0.1 * np.mean(improvements)
        above_threshold = [i for i, avg in enumerate(moving_avg) if avg >= threshold]
        
        if not above_threshold:
            optimal_start, optimal_end = 0, 0
        else:
            optimal_start = above_threshold[0]
            optimal_end = above_threshold[-1] + 1
        
        results[aspect] = {
            "optimal_range_start": optimal_start,
            "optimal_range_end": optimal_end,
            "score_at_start": aspect_scores[optimal_start],
            "score_at_end": aspect_scores[optimal_end] if optimal_end < len(aspect_scores) else aspect_scores[-1],
            "improvement_in_range": sum(improvements[optimal_start:optimal_end])
        }
    
    return results
```


This function:
- Determines the most effective range of iterations for improvement
- Methodology:
  - Uses moving average of improvements to identify sustained progress
  - Finds optimal range where improvements are above a certain threshold

Benefits:
- Aids in fine-tuning the Chain of Density process
- Identifies the most productive iteration range for each aspect of summary quality

---

```python
@weave.op()
def find_optimal_score_range(scores):
    if len(scores) < 2:
        return {aspect: {"optimal_range_start": 0, "optimal_range_end": 0, "highest_score": 0, "improvement_in_range": 0} for aspect in ['relevance', 'technical_quality', 'conciseness']}
    
    aspects = ['relevance', 'technical_quality', 'conciseness']
    results = {}
    
    for aspect in aspects:
        aspect_scores = [s[aspect]['score'] for s in scores]
        improvements = [aspect_scores[i+1] - aspect_scores[i] for i in range(len(aspect_scores)-1)]
        
        highest_score = max(aspect_scores)
        highest_score_index = aspect_scores.index(highest_score)
        
        # Find the best range leading up to the highest score
        best_start = 0
        best_end = highest_score_index
        best_improvement = sum(improvements[:highest_score_index])
        
        for start in range(highest_score_index):
            current_improvement = sum(improvements[start:highest_score_index])
            if current_improvement > best_improvement:
                best_start = start
                best_improvement = current_improvement
        
        results[aspect] = {
            "optimal_range_start": best_start,
            "optimal_range_end": highest_score_index,
            "score_at_start": aspect_scores[best_start],
            "score_at_end": highest_score,
            "improvement_in_range": best_improvement
        }
    
    return results
```


This function:
- Identifies the iteration range producing the highest quality summaries
- Process:
  - Finds the range leading up to the highest score for each aspect
  - Considers cumulative improvement within the range

Usefulness:
- Helps understand which iterations contribute most significantly to final summary quality
- Assists in optimizing the summarization process for maximum effectiveness

---

```python
@weave.op()
def process_iteration_summaries(model_output, instruction, model):
    iteration_scores = [score_summary(summary, f"Iteration Summary {i+1}", instruction, model)
                        for i, summary in enumerate(model_output["iteration_summaries"])]
    return {
        "long_tail_stats": calculate_long_tail_stats(iteration_scores),
        # Additional analyses can be added here if needed
    }
```


This function:
- Aggregates and analyzes scores across all summarization iterations
- Provides:
  - Holistic view of summary quality evolution throughout Chain of Density iterations
  - Comprehensive analysis of the iterative summarization approach

Value:
- Helps understand overall effectiveness of the iterative process
- Identifies trends in quality improvement across iterations

---

```python
@weave.op()
def quality_scorer(instruction, model_output, model="gpt-4o"):
    scores = {
        "iteration_summaries_analysis": {},
        "accumulated_summary": {},
        "final_summary": {}
    }

    try:
        # Process iteration summaries
        scores["iteration_summaries_analysis"] = process_iteration_summaries(model_output, instruction, model)

        # Score accumulated summary
        scores["accumulated_summary"] = score_summary(model_output["accumulated_summary"], "Accumulated Summary", instruction, model)

        # Score final summary
        scores["final_summary"] = score_summary(model_output["final_summary"], "Final Summary", instruction, model)

        # Flatten the scores dictionary for easier analysis
        flattened_scores = {}
        for key, value in scores.items():
            if isinstance(value, dict):
                flattened_scores[key] = flatten_dict(value)
            else:
                flattened_scores[key] = value
    
        scores = flatten_dict(flattened_scores)

    except Exception as e:
        print(f"Error in quality_scorer: {str(e)}")
        scores["error"] = str(e)

    return scores
```


This function:
- Serves as the main entry point for evaluating summarization quality
- Features:
  - Combines all previous metrics into a comprehensive evaluation
  - Analyzes iteration summaries, accumulated summary, and final summary

Benefits:
- Provides a detailed, multi-faceted assessment of the summarization pipeline's performance
- Offers insights into various aspects of summary quality
- Evaluates the effectiveness of the Chain of Density process as a whole

---

These evaluation metrics collectively provide a robust framework for assessing the quality and effectiveness of our Chain of Density summarization pipeline. By examining multiple aspects of summary quality across different stages of the process, we can gain valuable insights into the strengths and weaknesses of our approach, identify areas for improvement, and optimize the summarization process for maximum effectiveness.

## Running the Evaluation

Finally, we set up and run the evaluation:

![Evaluation Setup](./media/eval_comparison.gif)

```python
models = [
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620"
]

evaluation = weave.Evaluation(dataset=dataset, scorers=[quality_scorer])
for model in models:
    arxiv_chain_of_density_pipeline = ArxivChainOfDensityPipeline(model=model, density_iterations=8)
    await evaluation.evaluate(arxiv_chain_of_density_pipeline)
```

This code sets up a Weave Evaluation object and runs the evaluation for each model in our list.

## Optional: Advanced Chunking Technique

<details>
<summary>Click to Expand</summary>

The cookbook also includes an optional section on an advanced chunking technique to handle longer documents more effectively:

### Chunking

1. `chunk_text`: Splits the input text into manageable chunks, handling special cases like image descriptions.

```python
@weave.op()
def chunk_text(text, chunk_size):
    chunks = []
    current_chunk = ""
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # If adding this line would exceed the chunk size, start a new chunk
        if len(current_chunk) + len(line) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        current_chunk += line + "\n"
        
        # Special handling for image descriptions
        if line.startswith("[Image Descriptions for page"):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Collect all lines of the image description
            image_descriptions = line + "\n"
            i += 1
            while i < len(lines) and not lines[i].startswith("[END OF IMAGE DESCRIPTIONS]"):
                image_descriptions += lines[i] + "\n"
                i += 1
            if i < len(lines):
                image_descriptions += lines[i] + "\n"
            
            # Add the entire image description as a separate chunk
            chunks.append(image_descriptions.strip())
            current_chunk = ""
        else:
            i += 1
    
    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Combine smaller chunks to reach the desired chunk size
    combined_chunks = []
    current_combined_chunk = ""
    for chunk in chunks:
        if len(current_combined_chunk) + len(chunk) <= chunk_size:
            current_combined_chunk += chunk + "\n\n"
        else:
            if current_combined_chunk:
                combined_chunks.append(current_combined_chunk.strip())
            current_combined_chunk = chunk + "\n\n"
    
    if current_combined_chunk:
        combined_chunks.append(current_combined_chunk.strip())

    return combined_chunks
```

2. `summarize_chunk`: Summarizes an individual chunk, focusing on the given instruction and incorporating previous summary information.

```python
@weave.op()
def summarize_chunk(chunk, instruction, current_summary="", iteration=1, model="claude-3-5-sonnet-20240620"):
    # Construct a prompt for summarizing the chunk
    prompt = f"""Current summary:
    {current_summary}

    New information:
    {chunk}

    Instruction to focus on: {instruction}

    Iteration: {iteration}

    Create an extremely dense, highly technical summary that specifically addresses the given instruction. Follow these steps:

    1. Identify 3-5 key technical points from the new information that are directly relevant to the instruction, prioritizing:
    - Novel methodologies or algorithms related to the instruction
    - Specific quantitative results or metrics that address the instruction
    - Detailed experimental setups or parameters pertinent to the instruction
    - Precise definitions of domain-specific concepts mentioned in the instruction
    - Critical limitations or assumptions in the research that affect the instruction

    1. Integrate these points with the current summary, ensuring:
    - Direct relevance to the instruction at hand
    - No redundancy or oversimplification
    - Preservation of technical nuances and complexities specific to the instruction
    - Inclusion of relevant equations, formulas, or mathematical notations that help address the instruction
    - Accurate representation of statistical significance and error margins for instruction-related data

    1. Rephrase the combined information to maximize information density while maintaining focus on the instruction:
    - Use domain-specific terminology and jargon without simplification, as relevant to the instruction
    - Maintain the level of detail expected in a PhD-level discourse on the specific topic of the instruction
    - Incorporate precise citations or references where applicable to support the response
    - Preserve any conflicting viewpoints or ongoing debates in the field that relate to the instruction

    1. With each iteration, aim to increase information density by 30-40% without sacrificing technical accuracy or critical details that address the instruction.

    2. Ensure the summary includes instruction-specific:
    - Methodological details (e.g., exact algorithms, parameter settings) that are crucial to addressing the instruction
    - Precise quantitative results with appropriate units and error bounds that directly relate to the instruction
    - Detailed descriptions of novel techniques or approaches that are key to addressing the instruction
    - Critical analysis of strengths and limitations in the research as they pertain to the instruction

    Produce a summary that is significantly more information-dense and technically precise than the previous one, while remaining laser-focused on addressing the given instruction. Use language appropriate for a highly specialized audience in the field."""

    # Use the Anthropic API to generate the summary
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

3. `summarize_chunk_summaries`: Combines summaries from multiple chunks into a coherent whole.

```python
@weave.op()
def summarize_chunk_summaries(instruction, current_summary, chunk_summaries, model="claude-3-opus-20240229"):
    # Construct a prompt for combining chunk summaries
    prompt = f"""Given this current summary:

    {current_summary}

    And these chunk summaries:

    {' '.join(chunk_summaries)}

    And this instruction to focus on:

    {instruction}

    Create an extremely dense, final summary that refines the current summary by incorporating key information from the chunk summaries, while specifically addressing the given instruction. Follow these guidelines:

    1. Integrate the most relevant and important information from the chunk summaries into the current summary.
    2. Ensure all key technical content from both the current summary and chunk summaries that relates to the instruction is retained.
    3. Aim to reduce overall length by 30-40% while increasing information density.
    4. Prioritize highly specific methodologies, algorithms, metrics, and findings that directly address the instruction.
    5. Preserve precise quantitative data, including statistical significance and error margins where applicable and relevant to the instruction.
    6. Maintain the use of domain-specific terminology and technical jargon pertinent to the instruction.
    7. Use compact phrasing and remove any remaining non-essential information that doesn't directly contribute to addressing the instruction.
    8. If relevant to the instruction, include brief mentions of limitations, assumptions, or conflicting viewpoints from across all summaries.
    9. Optimize for information density while maintaining coherence for an expert audience, always keeping the focus on the given instruction.

    The final summary should be a highly concentrated, technical distillation of all provided summaries that specifically addresses the given instruction, suitable for specialists in the field."""

    # Use the Anthropic API to generate the combined summary
    return anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text
```

4. `summarize_chunk_iteration`: Manages the process of summarizing all chunks in a single iteration.

```python
@weave.op()
def summarize_chunk_iteration(chunks, instruction, current_summary, iteration, model):
    chunk_summaries = []
    # Summarize each chunk individually
    for i, chunk in enumerate(chunks, 1):
        current_summary = summarize_chunk(chunk, instruction, current_summary, iteration, model)
        chunk_summaries.append(current_summary)
        print(f"Iteration {iteration}, Chunk {i}:\n{current_summary}\n")
    # Combine all chunk summaries into a single summary
    current_summary = summarize_chunk_summaries(instruction, current_summary, chunk_summaries, model)
    print(f"Iteration {iteration}, Final Summary:\n{current_summary}\n")
    return current_summary, chunk_summaries
```

5. `iterative_chunk_summarization`: Performs multiple iterations of chunk-based summarization.

```python
@weave.op()
def iterative_chunk_summarization(chunks, instruction, current_summary, chunk_iterations, model):
    chunk_iteration_summaries = []
    chunk_summaries = []
    # Perform multiple iterations of chunk summarization
    for iteration in range(1, chunk_iterations + 1):
        current_summary, iteration_chunk_summaries = summarize_chunk_iteration(chunks, instruction, current_summary, iteration, model)
        chunk_iteration_summaries.append(current_summary)
        chunk_summaries.append(iteration_chunk_summaries)
    return current_summary, chunk_iteration_summaries, chunk_summaries
```

6. `chain_of_density_summarization`: Orchestrates the entire summarization process, including both chunk-based and density-based summarization steps.



```python
@weave.op()
def chain_of_density_summarization(instruction, text, model="claude-3-5-sonnet-20240620", chunk_size=8192, chunk_iterations=2, density_iterations=2):
    # Split the text into chunks
    chunks = chunk_text(text, chunk_size)
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

    # Perform chunk-based summarization
    current_summary, chunk_iteration_summaries, chunk_summaries = iterative_chunk_summarization(chunks, instruction, "", chunk_iterations, model)
    
    # Perform final density-based summarization
    current_summary, iteration_summaries = iterative_density_summarization(instruction, current_summary, density_iterations, model)
    final_summary_text = final_summary(instruction, current_summary, model)
    print(f"Final Summary:\n{final_summary_text}\n")

    # Return all intermediate and final results
    return {
        "final_summary": final_summary_text,
        "accumulated_summary": current_summary,
        "iteration_summaries": iteration_summaries,
        "chunk_iteration_summaries": chunk_iteration_summaries,
        "chunk_summaries": chunk_summaries
    }
```

This advanced chunking technique allows for more effective handling of longer documents, potentially improving the quality and comprehensiveness of the final summary. 

### Model Evaluation

> Note that the `ArxivChainOfDensityPipeline` class stays identical as the same `chain_of_density_summarization` function is replaced.

## Advanced Evaluation Metrics

To thoroughly assess the quality and effectiveness of our Chain of Density summarization pipeline, we implement a set of advanced evaluation metrics. These metrics provide a comprehensive analysis of the summarization process, taking into account both the chunk-based approach and the overall summary quality.

### Processing Chunk Summaries

The `process_chunk_summaries` function evaluates the quality of individual chunk summaries:

```python
def process_chunk_summaries(model_output, instruction, model):
    scores = {}
    for i, chunk_list in enumerate(model_output["chunk_summaries"]):
        chunk_summary_scores = []
        for j, summary in enumerate(chunk_list):
            chunk_summary_score = score_summary(summary, f"Chunk Summary {i+1}.{j+1}", instruction, model)
            chunk_summary_scores.append(chunk_summary_score)
        
        scores[f"chunk_summaries_analysis_{i+1}"] = {
            "long_tail_stats": calculate_long_tail_stats(chunk_summary_scores),
            "iteration_impact": analyze_iteration_impact(chunk_summary_scores),
            "optimal_improvement_range": find_optimal_improvement_range(chunk_summary_scores),
            "optimal_score_range": find_optimal_score_range(chunk_summary_scores)
        }
    return scores
```

This function:
- Scores each chunk summary individually
- Calculates various statistics for each chunk iteration, including long-tail stats, iteration impact, and optimal improvement ranges

### Processing Chunk Iteration Summaries

The `process_chunk_iteration_summaries` function evaluates the quality of summaries produced after each chunk iteration:

```python
def process_chunk_iteration_summaries(model_output, instruction, model):
    chunk_iteration_scores = [
        score_summary(summary, f"Chunk Iteration Summary {i+1}", instruction, model)
        for i, summary in enumerate(model_output["chunk_iteration_summaries"])
    ]
    
    return {
        "long_tail_stats": calculate_long_tail_stats(chunk_iteration_scores),
        "iteration_impact": analyze_iteration_impact(chunk_iteration_scores),
        "optimal_improvement_range": find_optimal_improvement_range(chunk_iteration_scores),
        "optimal_score_range": find_optimal_score_range(chunk_iteration_scores)
    }
```

This function:
- Scores each chunk iteration summary
- Calculates aggregate statistics across all chunk iterations

### Quality Scorer

The `quality_scorer` function serves as the main entry point for our evaluation process:

```python
@weave.op()
def quality_scorer(instruction, model_output, model="gpt-4o"):
    scores = {
        "chunk_summaries_analysis": {},
        "chunk_iteration_summaries_analysis": {},
        "iteration_summaries_analysis": {},
        "accumulated_summary": {},
        "final_summary": {}
    }

    try:
        chunk_summaries_scores = process_chunk_summaries(model_output, instruction, model)
        scores.update(chunk_summaries_scores)

        scores["chunk_iteration_summaries_analysis"] = process_chunk_iteration_summaries(model_output, instruction, model)
        scores["iteration_summaries_analysis"] = process_iteration_summaries(model_output, instruction, model)
        scores["accumulated_summary"] = score_summary(model_output["accumulated_summary"], "Accumulated Summary", instruction, model)
        scores["final_summary"] = score_summary(model_output["final_summary"], "Final Summary", instruction, model)

        flattened_scores = {}
        for key, value in scores.items():
            if isinstance(value, dict):
                flattened_scores[key] = flatten_dict(value)
            else:
                flattened_scores[key] = value
    
        scores = flatten_dict(flattened_scores)

    except Exception as e:
        print(f"Error in quality_scorer: {str(e)}")
        scores["error"] = str(e)

    return scores
```

This function:
- Orchestrates the entire evaluation process
- Processes and scores chunk summaries, chunk iteration summaries, and the final summary
- Flattens the nested score dictionary for easier analysis
- Handles any errors that occur during the scoring process

By implementing these advanced evaluation metrics, we can gain deep insights into the performance of our Chain of Density summarization pipeline at various stages of the process. This allows us to identify areas for improvement and optimize our approach for maximum effectiveness.

</details>

## Conclusion

This cookbook has demonstrated the implementation of an advanced AI-powered summarization bot using the Chain of Density technique. By leveraging Anthropic's Claude API, the Arxiv API, and Weave for experiment tracking, we've created a powerful tool for generating concise, information-dense summaries of scientific papers.

Key takeaways:
1. The Chain of Density technique allows for iterative refinement of summaries, increasing information density while maintaining relevance to specific instructions.
2. Our implementation handles both textual content and visual elements (images and vector graphics) from PDF papers, ensuring comprehensive coverage of research content.
3. The optional advanced chunking technique enables effective processing of longer documents, improving summary quality for extensive research papers.
4. Robust evaluation metrics provide insights into the summarization process, allowing for continuous improvement and optimization.

Potential applications:
- Rapid literature review for researchers
- Automated creation of paper abstracts or extended summaries
- Assisting in the peer review process by providing concise overviews of submissions
- Enhancing search and discovery of relevant research papers

By combining state-of-the-art language models with carefully crafted prompts and evaluation techniques, this summarization pipeline demonstrates the potential for AI to significantly accelerate and enhance scientific research processes.
