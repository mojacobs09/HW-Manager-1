import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("My Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

openai_api_key = st.secrets.OPENAI_API_KEY

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Summary choice (move to top with other inputs)
st.sidebar.header(":red[Summary Type]") 
summary_type = st.sidebar.selectbox(
    "Choose a summary type",
    ('Summarize in 100 words', 'Summarize in 2 connecting paragraphs', 'Summarize in 5 bullet points')
)

# Check box to select the higher model
use_advance = st.checkbox('Use Advanced Model')
if use_advance:
    model_name = 'gpt-4'
else: 
    model_name = "gpt-3.5-turbo"

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)

# Ask the user for a question via `st.text_area`.
question = st.text_area("Ask a question about the document:")

# Process the uploaded file and question.
if uploaded_file and question:
    document = uploaded_file.read().decode()
    
    # Generate an answer using the OpenAI API.
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n{summary_type}. Then answer this question: {question}",
        }
    ]
    
    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
    )
    
    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)
