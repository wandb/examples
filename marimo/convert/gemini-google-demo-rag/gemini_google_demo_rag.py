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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/gemini/google-demo-rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{gemini-weave-rag-demo} -->
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install google-generativeai weave -qqU
    return


@app.cell
def _():
    import google.generativeai as genai
    import weave

    return genai, weave


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up your Google API key and log into W&B Weave

    To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.
    """)
    return


@app.cell
def _(genai):
    from google.colab import userdata
    GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    return


@app.cell
def _():
    import asyncio
    from weave import Model, Evaluation, Dataset
    import numpy as np

    return Model, np


@app.cell
def _(weave):
    # We call init to begin capturing data in the project, intro-example.
    weave.init("prompt-eng/gemini-rag")
    return


@app.cell
def _(genai, np):
    from google.api_core import retry
    emb_model = 'models/embedding-001'

    def make_embed_text_fn(model):
        @retry.Retry(timeout=300.0)
        def embed_fn(text: str) -> list[float]:
            embedding = genai.embed_content(model=model,
                                            content=text,
                                            task_type="retrieval_document")['embedding']
            return np.array(embedding)
        return embed_fn

    return emb_model, make_embed_text_fn


@app.cell
def _(emb_model, make_embed_text_fn, weave):
    @weave.op()
    def docs_to_embeddings(docs: list) -> list:
        # Convert documents to embeddings
        document_embeddings = []
        emb_fn = make_embed_text_fn(emb_model)
        for doc in docs:
            emb = emb_fn(doc)
            document_embeddings.append(emb)
        return document_embeddings

    return (docs_to_embeddings,)


@app.cell
def _(emb_model, genai, np, weave):
    @weave.op()
    def get_most_relevant_document(query, docs, document_embeddings):
        # Convert query to embedding
        query_embedding = genai.embed_content(model=emb_model,
                                        content=query,
                                        task_type="retrieval_query")['embedding']
        # Compute cosine similarity
        similarities = np.dot(np.stack(document_embeddings), query_embedding)
        # Get the index of the most similar document
        most_relevant_doc_index = np.argmax(similarities)
        return docs[most_relevant_doc_index]

    return (get_most_relevant_document,)


@app.cell
def _(Model, genai, get_most_relevant_document, weave):
    # define the Morgan Stanley Research RAG Model
    class MSResearchRAGModel(Model):
         system_message: str
         model_name: str = 'models/gemini-pro'

         @weave.op()
         def predict(self, question: str, docs: list, add_context: bool) -> dict:
             model = genai.GenerativeModel(self.model_name)

             RAG_Context = ""
             # Retrieve the embeddings artifact
             embeddings = weave.ref("MSRAG_Embeddings").get()

             if add_context:
                 # Using Google Embeddings, get the relevant document for context
                 RAG_Context = get_most_relevant_document(question, docs, embeddings)

             query = f"""Use the following information to answer the subsequent question. If the answer cannot be found, write "I don't know."

                     Context from Morgan Stanley Research:
                     \"\"\"
                     {RAG_Context}
                     \"\"\"

                     Question: {question}"""
             prompt = f'{self.system_message}\n\n{query}'
             response = model.generate_content(prompt)     
             model_output = response.text
         
             return model_output

    return (MSResearchRAGModel,)


@app.cell
def _(MSResearchRAGModel, docs_to_embeddings, weave):
    model = MSResearchRAGModel(
         system_message="You are an expert in finance and answer questions related to finance, financial services, and financial markets. When responding based on provided information, be sure to cite the source."
     )

    contexts = [
         f"""Morgan Stanley has moved in new market managers on the East and West Coasts as part of changes that sent some of other management veterans into new roles, according to two sources.

     On the West Coast, Ken Sullivan, a 37-year-industry veteran who came to Morgan Stanley five years ago from RBC Wealth Management, has assumed an expanded role as market executive for a consolidated Beverly Hills and Los Angeles market, according to a source.

     Meanwhile, Greg Laetsch, a 44-year industry veteran who had been the complex manager in Los Angeles for the last 15 years, has moved to a non-producing senior advisor role to the LA market for Morgan Stanley,  according to the same source.

     On the East Coast, Morgan Stanley hired Nikolas Totaro, a 19-year industry veteran, from Merrill Lynch, where he had worked for 14 years and had been most recently a market executive in Greenwich, Connecticut. Totaro will be a market manager reporting to John Palazzetti in the Midtown Wealth Management Center in Manhattan, according to the same source.

     Totaro is replacing Bill DeMatteo, a 21-year industry veteran who spent the last 14 years at Morgan Stanley, and who has returned to full-time production. DeMatteo has joined the Continuum Group at Morgan Stanley, which Barron’s ranked 20th among on its 2022 Top 100 Private Wealth Management Teams and listed as managing $7.2 billion in client assets.

     “His extensive 17 years of management experience at Morgan Stanley will be instrumental in shaping our approach to wealth management, fostering client relationships, and steering the team towards sustained growth and success,” Scott Siegel, leader of the Continuum Group, wrote on LinkedIn.

     Totaro and Laestch did not respond immediately to requests for comments sent through LinkedIn. Sullivan did not respond immediately to an emailed request. Both Morgan Stanley and Merrill spokespersons declined to comment about the changes.

     Totaro’s former Southern Connecticut market at Merrill included over 325 advisors and support staff across six offices in Greenwich, Stamford, Darien, Westport, Fairfield, and New Canaan, according to his LinkedIn profile.

     Separately, a former Raymond James Financial divisional director has joined Janney Montgomery Scott in Ponte Vedra Beach, Florida. Tom M. Galvin, who has spent the last 25 years with Raymond James, joins Janney as a complex director, according to an announcement.

     Galvin had most recently worked as a divisional director for Raymond James & Associates’ Southern Division. The firm consolidated the territory as part of a reorganization that took effect December 1. Galvin’s registration with Raymond James ended November 8, according to BrokerCheck.

     During his career, Galvin has held a range of branch and complex management roles in the North Atlantic and along the East Coast, according to his LinkedIn profile.

     “We’re looking forward to his experience and strong industry relationships as we continue to expand our team and geographic footprint,” stated Janney’s Florida Regional Director Frank Amigo, who joined from Raymond James in 2017.

     Galvin started his career in 1995 with RBC predecessor firm J. B. Hanauer & Co. and joined Raymond James two years later, according to BrokerCheck. He did not immediately respond to a request for comment sent through social media.""",
         f"""Don’t Count on a March Rate Cut - Raise Rates
     Inflation will be stickier than expected, delaying the start of long-awaited interest rate cuts.
     Investors expecting a rate cut in March may be disappointed.
     Six-month core consumer price inflation is likely to increase in the first quarter, prompting the Fed to watch and wait.
     Unless there is an unexpectedly sharp economic downturn or weakening in the labor market, rate cuts are more likely to begin in June.

     Investors betting that the U.S. Federal Reserve will begin trimming interest rates in the first quarter of 2024 may be in for a disappointment.

     After the Fed’s December meeting, market expectations for a March rate cut jumped to surprising heights. Markets are currently putting a 75% chance, approximately, on rate cuts beginning in March. However, Morgan Stanley Research forecasts indicate that cuts are unlikely to come before June.

     Central bank policymakers have likewise pushed back on investors’ expectations. As Federal Reserve Chairman Jerome Powell said in December 2023, when it comes to inflation, “No one is declaring victory. That would be premature.”1

     Here’s why we still expect that rates are likely to hold steady until the middle of 2024.

     Inflation Outlook
     A renewed uptick in core consumer prices is likely in the first quarter, as prices for services remain elevated, led by healthcare, housing and car insurance. Additionally, in monitoring inflation, the Fed will be watching the six-month average—which means that weaker inflation numbers from summer 2023 will drop out of the comparison window. Although annual inflation rates should continue to decline, the six-month gauge could nudge higher, to 2.4% in January and 2.69% in February.

     Labor markets have also proven resilient, giving Fed policymakers room to watch and wait.

     Data-Driven Expectations
     Data is critical to the Fed’s decisions and Morgan Stanley’s forecasts, and both could change as new information emerges. At the March policy meeting, the Fed will have only data from January and February in hand, which likely won’t provide enough information for the central bank to be ready to announce a rate cut. The Fed is likely to hold rates steady in March unless nonfarm payrolls add fewer than 50,000 jobs in February and core prices gain less than 0.2% month-over-month. However, unexpected swings in employment and consumer prices, or a marked change in financial conditions or labor force participation, could trigger a cut earlier than we anticipate.

     There are scenarios in which the Fed could cut rates before June, including: a pronounced deterioration in credit conditions, signs of a sharp economic downturn, or slower-than-expected job growth coupled with weak inflation. Weaker inflation and payrolls could bolster the chances of a May rate cut especially.

    When trying to assess timing, statements from Fed insiders are good indicators because they tend to communicate premeditated changes in policy well in advance. If the Fed plans to hold rates steady in March, they might emphasize patience, or talk about inflation remaining elevated. If they’re considering a cut, their language will shift, and they may begin to say that a change in policy may be appropriate “in coming meetings,” “in coming months” or even “soon.” But a long heads up is not guaranteed.
    https://www.morganstanley.com/ideas/fed-rate-cuts-2024
    """,
        f"""What Global Turmoil Could Mean for Investors
    Weighing the investment impacts of global conflict and geopolitical tensions to international trade, oil prices and China equities.
    Morgan Stanley Research expects cargo shipping to remain robust despite Red Sea disruption.
    Crude oil shipments and oil prices should see limited negative impact from regional conflict.
    Long-term trends could bring growth in Japan and India.
    In a multipolar world, competition for global power is increasingly leading countries to protect their military and economic interests by erecting new barriers to cross-border commerce in key industries such as technology and renewable energy. As geopolitics and national security are to a growing degree driving how goods flow and where big capital investments are made, it’s that much more crucial for investors to know how to pick through a dizzying amount of information and focus on what’s relevant. But it’s hard to do with a seemingly endless series of alerts lighting up your phone.

    In particular, potential ripples from U.S.-China relations as well as U.S. military involvement in the Middle East could be important for investors. Morgan Stanley Research pared back the headlines and market noise to home in on three key takeaways.

    Gauging Red Sea Disruption
    Commercial cargo ships in the Red Sea handle about 12% of global trade. Attacks on these ships by Houthi militants, and ongoing U.S. military strikes to quell the disruption, have raised concerns that supply chains could see pandemic-type disruption—and a corresponding spike in inflation.

    However, my colleagues and I expect the flow of container ships to remain robust, even if that flow is redirected to avoid the Red Sea, which serves as an outlet for vessels coming out of the Suez Canal. Although there has been a recent 200% surge in freight rates, there have not been fundamental cost increases for shipping. Additionally, there’s currently a surplus of container ships. Lengthy reroutes around the Southern tip of Africa by carriers to avoid the conflict zone may cause delays, but they should have minimal impact to inflation in Europe. The risks to the U.S. retail sector should be similarly manageable.

    Resilience in Oil and the Countries That Produce it
    The Middle East is responsible for supplying and producing the majority of the world’s oil, so escalating conflict in the region naturally puts pressure on energy supply, as well the economic growth of relevant countries. However, the threat of weaker growth, higher inflation and erosion of support from allies offer these countries an incentive to contain the conflict. As a result, there’s unlikely to be negative impact to the debt of oil-producing countries in the region. Crude oil shipments should also see limited impacts, though oil prices could spike and European oil refiners, in particular, could face pressure if disruption in the Strait of Hormuz, which traffics about a fifth of oil supplies daily, accelerates.

    Opportunities in Asia Emerging in Japan and India
    China has significant work to do to retool its economic engine away from property, infrastructure and debt, leading Morgan Stanley economists to predict gross-domestic product growth of 4.2% for 2024 (below the government’s 5% target), slowing to 1.7% from 2025 to 2027. As a result, China’s relatively low equity market valuation still faces challenges, including risks such as U.S. policy restricting future investment. But elsewhere in Asia—particularly in standouts Japan and India—positive long-term trends should drive markets higher. These include fiscal rebalancing, increased digitalization and increasing shifts of manufacturing and supply hubs in a multipolar world.

    For a deeper insights and analysis, ask your Morgan Stanley Representative or Financial Advisor for the full report, “Paying Attention to Global Tension.”
    https://www.morganstanley.com/ideas/geopolitical-risk-2024
    """,
        f"""What 'Edge AI' Means for Smartphones
    As generative artificial intelligence gets embedded in devices, consumers should see brand new features while smartphone manufacturers could see a sales lift.
    Advances in artificial intelligence are pushing computing from the cloud directly onto consumer devices, such as smartphones, notebooks, wearables, automobiles and drones.
    This trend is expected to drive smartphone sales during the next two years, reversing a slowdown that began in 2021.
    Consumers can expect new features, such as touch-free control of their phones, desktop-quality gaming and real-time photo retouching.
    As the adoption of generative artificial intelligence accelerates, more computing will be done in the hands of end users—literally. Increasingly, AI will be embedded in consumer devices such as smartphones, notebooks, wearables, automobiles and drones, creating new opportunities and challenges for the manufacturers of these devices.

    Generative AI’s phenomenal capabilities are power-intensive. So far, the processing needed to run sophisticated, mainstream generative AI models can only take place in the cloud. While the cloud will remain the foundation of AI infrastructure, more AI applications, functions and services require faster or more secure computing closer to the consumer. “That’s driving the need for AI algorithms that run locally on the devices rather than on a centralized cloud—or what’s known as the AI at the Edge,” says Ed Stanley, Morgan Stanley’s Head of Thematic Research in London.

    By 2025, Edge AI will be responsible for half of all enterprise data created, according to an estimate by technology market researcher Gartner Inc. While there are many hurdles to reaching commercial viability, the opportunity to tap into 30 billion devices could reduce cost, increase personalization, and improve security and privacy. In addition, faster algorithms on the Edge can reduce latency (i.e., the lag in an app’s response time as it communicates with the cloud).

    “If 2023 was the year of generative AI, 2024 could be the year the technology moves to the Edge,” says Stanley. “We think this trend will pick up steam in 2024, and along with it, opportunities for hardware makers and component suppliers that can help put AI directly into consumers' hands.”

    New Smartphones Lead the Charge
    Smartphones currently on the market rely on traditional processors and cloud-based computing, and the only AI-enabled programs are features like face recognition, voice assist and low-light photography. Device sales have slowed in recent years, and many investors expect that smartphones will follow the trajectory of personal computers, with multi-year downturns as consumers hold onto their devices for longer due to lack of new features, sensitivity to pricing and other factors.

    But thanks in part to Edge AI, Morgan Stanley analysts think the smartphone market is poised for an upswing and predict that shipments, which have slowed since 2021, will rise by 3.9% this year and 4.4% next year.

    “Given the size of the smartphone market and consumers’ familiarity with them, it makes sense that they will lead the way in bringing AI to the Edge,” says Morgan Stanley’s U.S. Hardware analyst Erik Woodring. “This year should bring a rollout of generative AI-enabled operating systems, as well as next-generation devices and voice assistants that could spur a cycle of smartphone upgrades.”

    However, the move to the Edge will require new smartphone capabilities, especially to improve battery life, power consumption, processing speed and memory. Manufacturers with the strongest brands and balance sheets are best positioned to take the lead in the hardware arms race.

    Killer Apps
    In addition to hardware, AI itself continues to evolve. New generations of AI models are designed be more flexible and adaptable for a wide range of uses, including Edge devices. Other beneficiaries include smartphone memory players, integrated circuit makers and camera parts suppliers that support new AI applications.

    What can you expect from your phone in the next year?

    “Always-sensing cameras” that automatically activate or lock the screen by detecting if the user is looking at it without the need to touch the screen. This feature could also automatically launch applications such as online payment and food ordering by detecting bar codes.

    Gesture controls for when the user is unable to hold their devices, such as while cooking or exercising.

    Desktop-quality gaming experiences that offer ultra-realistic graphics with cinematic detail, all with smoother interactions and blazing-fast response times.

    Professional-level photography in which image processors enhance photos and video in real time by recognizing each element in a frame—faces, hair, glasses, objects—and fine tune each, eliminating the need for retouching later.

    Smarter voice assistance that is more responsive and tuned the user’s voice and speech patterns, and can launch or suggest apps based on auditory clues.

    “With Edge AI becoming part of everyday life, we see significant opportunities ahead as new hardware provides a platform for developers to create ground-breaking generative AI apps, which could trigger a new hardware product cycle that liftsservices sales,” says Woodring.

    For deeper insights and analysis, ask your Morgan Stanley Representative or Financial Advisor for the full reports, “Tech Diffusion: Edge AI—Growing Impetus” (Nov. 7, 2023), “Edging Into a Smartphone Upcycle” (Nov. 9, 2023) and “Edge AI: Product Releases on Track, But Where Are Killer Apps?”
    https://www.morganstanley.com/ideas/edge-ai-devices-diffusion""",
    ]

    questions = [
        "Can you summarize the latest changes to Morgan Stanley market managers?",
        "When will the fed lower rates?",
        "What are the top market risks?",
        "How will AI impact the smartphone market?",
    ]

    # Calculate the document embeddings and store in weave
    document_embeddings = docs_to_embeddings(contexts)
    embeddings_ref = weave.publish(document_embeddings, "MSRAG_Embeddings")

    for i in range(0, len(questions)):
        # Using Google Embeddings
        model.predict(questions[i], contexts, True)

    # Not using Google Embeddings
    model.predict(questions[1], contexts, False)
    return


if __name__ == "__main__":
    app.run()
