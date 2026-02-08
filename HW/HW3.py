import streamlit as st
from openai import OpenAI
import requests 
import anthropic
from bs4 import BeautifulSoup

st.title('My LAB3 Question Answering Bot')

st.write('''
**How this chatbot works:**
- Enter up to 2 URLs to provide context for the conversation
- Choose your preferred LLM vendor and model
- **Conversation Memory:** This bot uses a buffer of 6 messages (3 user-assistant exchanges)
- The system prompt with URL content is never discarded
''')

def read_url_content(url):
    try: 
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()[:3000]
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Buffer function - keeps last 6 messages + system prompt
def trim_messages(messages, max_messages=6):
    """Keep system prompt + last 6 messages (3 user-assistant exchanges)"""
    system_msgs = [msg for msg in messages if msg['role'] == 'system']
    other_msgs = [msg for msg in messages if msg['role'] != 'system']
    
    trimmed = other_msgs[-max_messages:] if len(other_msgs) > max_messages else other_msgs
    
    return system_msgs + trimmed

# Sidebar - LLM Settings
st.sidebar.subheader('LLM Settings')
llm_vendor = st.sidebar.selectbox('Which Vendor?', ('OpenAI', 'Anthropic'))

if st.session_state.current_vendor != llm_vendor:
    if 'client' in st.session_state:
        del st.session_state.client
    if 'messages' in st.session_state:
        del st.session_state.messages
    st.session_state.current_vendor = llm_vendor



if llm_vendor == 'OpenAI':
    openAI_model = st.sidebar.selectbox('Which Model?', ('mini', 'premium'))
    if openAI_model == 'mini':
        model_to_use = 'gpt-5-mini-2025-08-07'
    else:
        model_to_use = 'gpt-5-2025-08-07'
else:
    anthropic_model = st.sidebar.selectbox('Which Model?', ('haiku', 'premium'))
    if anthropic_model == 'haiku':
        model_to_use = 'claude-haiku-4-5-20251001'
    else:
        model_to_use = 'claude-sonnet-4-5-20250929'

# creating client 
if 'client' not in st.session_state:
    if llm_vendor == 'OpenAI':
        api_key = st.secrets['OPENAI_API_KEY']
        st.session_state.client = OpenAI(api_key=api_key)
        st.session_state.vendor = 'OpenAI'
    else:
        api_key = st.secrets['ANTHROPIC_API_KEY']
        st.session_state.client = anthropic.Anthropic(api_key=api_key)
        st.session_state.vendor = 'Anthropic'

# Sidebar - URL Input
st.sidebar.subheader('URL Input')
url1 = st.sidebar.text_input('URL 1 (optional):')
url2 = st.sidebar.text_input('URL 2 (optional):')

# Build system prompt with URL content
system_content = '''You are a helpful assistant that explains things simply for 10-year-olds.
After answering each question, ask "Do you want more info?"
If the user says yes, provide more detailed information and ask again.
If the user says no, ask "What else can I help you with?"'''

if url1:
    content1 = read_url_content(url1)
    if content1:
        system_content += f"\n\nContext from {url1}:\n{content1}"

if url2:
    content2 = read_url_content(url2)
    if content2:
        system_content += f"\n\nContext from {url2}:\n{content2}"



# Initialize messages with system prompt
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'system', 'content': system_content},
        {'role': 'assistant', 'content': 'How can I help you?'}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg['role'] != 'system':
        chat_msg = st.chat_message(msg['role'])
        chat_msg.write(msg['content'])

# Chat input
if prompt := st.chat_input('What is up?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Apply 6-message buffer
    messages_to_send = trim_messages(st.session_state.messages, max_messages=6)
    
    client = st.session_state.client
    
    # Handle different API formats for OpenAI vs Anthropic
    if llm_vendor == 'OpenAI':
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages_to_send,
            stream=True
        )
        
        with st.chat_message('assistant'):
            response = st.write_stream(stream)
    
    else:  # Anthropic
        # Anthropic needs system separate from messages
        system_msg = [msg for msg in messages_to_send if msg['role'] == 'system'][0]['content']
        other_msgs = [msg for msg in messages_to_send if msg['role'] != 'system']
        
        with st.chat_message('assistant'):
            with client.messages.stream(
                model=model_to_use,
                max_tokens=1024,
                system=system_msg,
                messages=other_msgs
            ) as stream:
                response = st.write_stream(stream.text_stream)
    
    st.session_state.messages.append({'role': 'assistant', 'content': response})