import pandas as pd
import streamlit as st

from io import StringIO

st.set_page_config(
    page_title="W&B Annotation",
    # page_icon="🧊",
    layout="wide",
)
st.markdown('''<style>
            a {
              color: #13A9BA !important;
              text-decoration: none;
              transition: .3s;
            }
            a:hover {
              color: #0097AB !important;
              text-decoration: none;
            }
            .st-emotion-cache-187vp62 p {
              margin-bottom: .5rem;
            }
            ol {
              padding-left: 1.5rem;
            }
            </style>''', unsafe_allow_html=True)

st.image("images/wandb-streamlit-logo.png")
st.title('Custom annotation of W&B Tables with the Streamlit data editor')
st.write('Summarization can be a critical but challenging language modeling task, with varying manual and automated approaches that prove hard to evaluate and compare. [Weights & Biases]("https://wandb.ai/site") helps machine learning practitioners log summary inputs and results from multiple experimental approaches and interrogate and evaluate those results effectively at scale. [Streamlit’s data editor]("https://docs.streamlit.io/library/api-reference/data/st.data_editor"), showcased in this application, helps teams responsible for annotating modeling results for interim and final assessments in language modeling pipelines interact with these results and revise them for downstream tasks (e.g., creating gold standard examples or fine-tuning).')
         
st.write('This application takes an input csv file of news articles and automated summaries using [BART]("https://huggingface.co/facebook/bart-large-cnn") and [BART-SAMSUM]("https://huggingface.co/philschmid/bart-large-cnn-samsum") and allows a user to smoothly evaluate summaries from multiple approaches. Specifically, a user can:')

st.markdown('''
            1. select whether an automated summary requires adjustment (***needs_revision** column*)
            2. enter a suggested summary approach from a drop-down menu (***approach** column*)
            3. enter a manual edition of the current summary (***manual_edition** column*)
            4. add optional comments for the revision (***comments** column*)

''')

st.write('Users can upload two separate files to compare experimental results from different approaches and annotate dynamically in-app.')

with st.expander("See additional resources"):
    
    st.link_button("Annotation Colab notebook", "https://colab.research.google.com/drive/133sV8VgY5wftiDpjIT3UnNcvBwTQzfSa?usp=sharing")
    st.link_button("W&B project for Tables integration", "https://wandb.ai/claire-boetticher/news_summarization?workspace=user-claire-boetticher")

# first dataframe with manual revision columns

uploaded_file = st.file_uploader("Choose a file for summary review (Approach 1)", key="bart")
if uploaded_file is not None:

    # Read csv as dataframe
    dataframe = pd.read_csv(uploaded_file)
    dataframe.rename(columns={'Unnamed': 'Row'}, inplace=True)

    # Reorder index from random sample row numbers - optional
    # dataframe.reset_index(drop=True)

    # Add empty column for free text comments
    dataframe['needs_revision'] = ''
    dataframe['approach'] = ''
    dataframe['manual_edition'] = ''
    dataframe['comments'] = ''
  
    # Display dataframe in app
    bart_edited_df = st.data_editor(dataframe, 
                               column_config={
                                   "needs_revision": st.column_config.CheckboxColumn(
                                       "needs_revision",
                                       help="Does the Summary column need to be changed?",
                                       width="medium",
                                       default=False,
                                       required=True,
                                   ),
                                   "approach": st.column_config.SelectboxColumn(
                                       "approach",
                                       help="Select approach for revising summary",
                                       width="medium",
                                       options=[
                                           "Manual edit",
                                           "Adjust model",
                                           "Other (add suggestion in Comments column)"
                                       ],
                                       required=False,
                                   ),
                                    "manual_edition": st.column_config.TextColumn(
                                       "manual_edition",
                                       help="Enter new summary",
                                       width="medium",
                                       required=False,
                                   ),
                                   "comments": st.column_config.TextColumn(
                                       "comments",
                                       help="Describe reasoning for editing the current assigned Summary value",
                                       width="large",
                                       required=False,
                                   ),
                               },
                               # disabled freezes columns so users cannot change the values
                               disabled=("articles", "bart_summaries", "source_word_count", "summary_word_count", "source_lexical_diversity", "summary_lexical_diversity"),
                               hide_index=True,
                               column_order=("articles", "bart_summaries", "needs_revision", "approach", "manual_edition", "comments", "source_word_count", "summary_word_count", "source_lexical_diversity", "summary_lexical_diversity"),
                               num_rows="dynamic"

    )

# second dataframe with different columns and metrics for evaluation
    
uploaded_file = st.file_uploader("Choose a file for summary review (Approach 2)", key="samsum")
if uploaded_file is not None:

    # Read csv as dataframe
    dataframe = pd.read_csv(uploaded_file)
    dataframe.rename(columns={'Unnamed': 'Row'}, inplace=True)

    # Reorder index from random sample row numbers - optional
    # dataframe.reset_index(drop=True)

    # Add empty column for free text comments
    dataframe['relevance'] = ''
    dataframe['coherence'] = ''
    dataframe['consistency'] = ''
    dataframe['fluency'] = ''
    dataframe['needs_revision'] = ''
    dataframe['approach'] = ''
    dataframe['manual_edition'] = ''
    dataframe['comments'] = ''
  
    # Display dataframe in app
    samsum_edited_df = st.data_editor(dataframe, 
                               column_config={
                                    "relevance": st.column_config.SelectboxColumn(
                                        "relevance (1-5)",
                                        help="select a relevance score from 1 (lowest) to 5 (highest)",
                                        width="medium",
                                        options=[
                                           "1",
                                           "2",
                                           "3",
                                           "4",
                                           "5"
                                       ],
                                    ),
                                    "coherence": st.column_config.SelectboxColumn(
                                        "coherence (1-5)",
                                        help="select a coherence score from 1 (lowest) to 5 (highest)",
                                        width="medium",
                                        options=[
                                           "1",
                                           "2",
                                           "3",
                                           "4",
                                           "5"
                                       ],
                                    ),
                                    "consistency": st.column_config.SelectboxColumn(
                                        "consistency (1-5)",
                                        help="select a consistency score from 1 (lowest) to 5 (highest)",
                                        width="medium",
                                        options=[
                                           "1",
                                           "2",
                                           "3",
                                           "4",
                                           "5"
                                       ],
                                    ),
                                    "fluency": st.column_config.SelectboxColumn(
                                        "fluency (1-5)",
                                        help="select a fluency score from 1 (lowest) to 5 (highest)",
                                        width="medium",
                                        options=[
                                           "1",
                                           "2",
                                           "3",
                                           "4",
                                           "5"
                                       ],
                                    ),
                                    "needs_revision": st.column_config.CheckboxColumn(
                                        "needs_revision",
                                        help="Does the summary column need to be changed?",
                                        width="medium",
                                        default=False,
                                        required=True,
                                    ),
                                    "manual_edition": st.column_config.TextColumn(
                                       "manual_edition",
                                       help="Enter new summary",
                                       width="medium",
                                       required=False,
                                   ),
                                   "comments": st.column_config.TextColumn(
                                       "comments",
                                       help="Describe reasoning for editing the current assigned Summary value",
                                       width="large",
                                       required=False,
                                   ),
                               },
                               # disabled freezes columns so users cannot change the values
                               disabled=("articles", "bart_samsum_summaries", "source_word_count", "summary_word_count", "source_lexical_diversity", "summary_lexical_diversity"),
                               hide_index=True,
                               column_order=("articles", "bart_samsum_summaries", "relevance", "coherence", "consistency", "fluency", "needs_revision", "manual_edition", "comments", "source_word_count", "summary_word_count", "source_lexical_diversity", "summary_lexical_diversity"),
                               num_rows="dynamic"

                )


## Add session state logic so that if Needs Revision is True, other columns are required
# import pandas as pd
# import streamlit as st

# data_df = pd.DataFrame(
#     {
#         "widgets": ["st.selectbox", "st.number_input", "st.text_area", "st.button"],
#         "favorite": [True, False, False, True],
#     }
# )

# st.data_editor(
#     data_df,
#     column_config={
#         "favorite": st.column_config.CheckboxColumn(
#             "Your favorite?",
#             help="Select your **favorite** widgets",
#             default=False,
#         )
#     },
#     disabled=["widgets"],
#     hide_index=True,
# )
