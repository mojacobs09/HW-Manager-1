import streamlit as st
from openai import OpenAI
import requests 
from bs4 import BeautifulSoup

def read_url_content(url):
    try: 
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

# Show title and description.
st.title("My Document question answering")
st.write(
    "Enter a URL below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Get API keys
openai_api_key = st.secrets.OPENAI_API_KEY
claude_api_key = st.secrets.CLAUDE_API_KEY

# Sidebar options
st.sidebar.header(":red[Summary Type]") 
summary_type = st.sidebar.selectbox(
    "Choose a summary type",
    ('Summarize in 100 words', 'Summarize in 2 connecting paragraphs', 'Summarize in 5 bullet points')
)

# LLM choice 
llm_type = st.sidebar.selectbox(
    'Select a LLM',
    ('OpenAI', 'Claude')
)

# Check box to select the higher model
use_advance = st.sidebar.checkbox('Use Advanced Model')

# Let the user enter a URL
url = st.text_input('Enter a URL:')

# Let the user choose their language
language_choice = st.selectbox(
    'Choose a language for your response',
    ('English', 'Spanish', 'Italian')
)

# Ask the user for a question via `st.text_area`.
question = st.text_area("Ask a question about the content:")

# Get document content from URL
document = None
if url:
    document = read_url_content(url)

# Process the document and question
if document and question:
    # Create the appropriate client based on user's choice
    if llm_type == 'OpenAI':
        client = OpenAI(api_key=openai_api_key)
        
        # Select model based on checkbox
        if use_advance:
            model_name = 'gpt-4'
        else: 
            model_name = "gpt-3.5-turbo"
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n{summary_type}. Then answer this question: {question}. Respond in {language_choice}.",
            }
        ]
        
        # Generate response
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        )
        
        # Stream the response
        st.write_stream(stream)
        
    else:  # Claude
        import anthropic
        client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Select model based on checkbox
        if use_advance:
            model_name = "claude-opus-4-20250514"
        else:
            model_name = "claude-3-5-sonnet-20241022"
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n{summary_type}. Then answer this question: {question}. Respond in {language_choice}.",
            }
        ]
        
        # Generate response
        stream = client.messages.stream(
            model=model_name,
            max_tokens=1024,
            messages=messages,
        )
        
        # Stream the response
        st.write_stream(stream)