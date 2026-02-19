import streamlit as st 
from openai import OpenAI
import sys
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(BASE_DIR, 'HW-4-su-org'))

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

def extract_text_from_html(html_path):
    '''
    Extracts text from an HTML file with error handling
    '''
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
    except Exception as e:
        st.error(f"Error reading HTML: {e}")
        return ""

# CHUNKING METHOD: Simple Split by Character Count
# This method splits each document into two equal halves based on character count. I chose this method because: It's simple and reliable for HTML documents of varying sizes and ensures each document contributes exactly 2 chunks as required
# It also maintains context better than splitting by arbitrary delimiters and works well for documents that don't have clear section breaks


def chunk_text(text, file_name):
    '''
    Splits text into 2 chunks of roughly equal size
    Returns list of tuples: (chunk_id, chunk_text)
    '''
    mid_point = len(text) // 2
    chunk_1 = text[:mid_point]
    chunk_2 = text[mid_point:]
    
    return [
        (f"{file_name}_chunk_1", chunk_1),
        (f"{file_name}_chunk_2", chunk_2)
    ]

def add_to_collection(collection, text, chunk_id, file_name):
    # creates an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small' 
    )
    embedding = response.data[0].embedding
    
    # add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids=[chunk_id],
        embeddings=[embedding],
        metadatas=[{"filename": file_name, "chunk_id": chunk_id}]
    )

def load_html_to_collection(folder_path, collection):
    html_files = list(Path(folder_path).glob('*.html'))
    print(f"Found {len(html_files)} HTML files: {html_files}")
    for html_file in html_files:
        text = extract_text_from_html(html_file)
        if text:
            chunks = chunk_text(text, html_file.name)
            for chunk_id, chunk_text in chunks:
                add_to_collection(collection, chunk_text, chunk_id, html_file.name)
    return True

# creating the vector database function 
def create_vector_db():
    # create Chroma Client
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')
    collection = chroma_client.get_or_create_collection('HW4Collection')
    
    # checking if collection is empty - only create if doesn't exist
    if collection.count() == 0:
        with st.spinner('Loading HTML files into collection...'):
            loaded = load_html_to_collection(FOLDER_PATH, collection)
            st.success(f'Loaded {collection.count()} document chunks!')
    return collection
    
if 'HW4_VectorDB' not in st.session_state:
    st.session_state.HW4_VectorDB = create_vector_db()

# Buffer function - keeps last 5 interactions (10 messages)
def trim_messages(messages, max_messages=10):
    """Keep system prompt + last 10 messages (5 user-assistant exchanges)"""
    system_msgs = [msg for msg in messages if msg['role'] == 'system']
    other_msgs = [msg for msg in messages if msg['role'] != 'system']
    
    trimmed = other_msgs[-max_messages:] if len(other_msgs) > max_messages else other_msgs
    
    return system_msgs + trimmed

# MAIN APP
st.title('HW 4: RAG Chatbot with HTML Documents')

st.write('''
**How this chatbot works:**
- This chatbot uses RAG (Retrieval Augmented Generation) with HTML documents
- Each document is chunked into 2 pieces for better retrieval
- Ask questions and the bot will search relevant document chunks to provide accurate answers
- The bot will clearly indicate when it's using information from the documents
- **Conversation Memory:** This bot stores the last 5 interactions (10 messages)
''')

# Initialize messages with system prompt
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'system', 'content': '''You are a helpful assistant that answers questions using provided context from HTML documents.

When answering questions:
- If you find relevant information in the provided documents, start your answer with "Based on the documents..." or "According to [document name]..."
- Clearly cite which document(s) you're using in your response
- If the documents don't contain relevant information, say "I don't have information about that in the available documents" and provide a general answer if appropriate
- Be concise but thorough in your responses'''},
        {'role': 'assistant', 'content': 'How can I help you? Ask me anything about the documents!'}
    ]

# Display chat messages
for msg in st.session_state.messages:
    if msg['role'] != 'system':
        chat_msg = st.chat_message(msg['role'])
        chat_msg.write(msg['content'])

# Chat input
if prompt := st.chat_input('What is your question?'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Get relevant context from vector database
    client = st.session_state.openai_client
    embedding_response = client.embeddings.create(
        input=prompt,
        model='text-embedding-3-small'
    )
    query_embedding = embedding_response.data[0].embedding
    
    # Query the vector database
    results = st.session_state.HW4_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    # Build context from retrieved documents
    context = ""
    for i in range(len(results['documents'][0])):
        doc_content = results['documents'][0][i][:1500]
        chunk_id = results['ids'][0][i]
        context += f"\n\n--- Chunk: {chunk_id} ---\n{doc_content}\n"
    
    # Prepare messages with context (last 5 interactions = 10 messages)
    messages_to_send = trim_messages(st.session_state.messages, max_messages=10)
    
    # Add context to system message for this request
    messages_with_context = []
    for msg in messages_to_send:
        if msg['role'] == 'system':
            messages_with_context.append({
                'role': 'system',
                'content': msg['content'] + f"\n\nHere are the relevant document chunks for context:\n{context}"
            })
        else:
            messages_with_context.append(msg)
    
    # Get response from OpenAI
    stream = client.chat.completions.create(
        model='gpt-5-2025-08-07',
        messages=messages_with_context,
        stream=True
    )
    
    with st.chat_message('assistant'):
        response = st.write_stream(stream)
    
    st.session_state.messages.append({'role': 'assistant', 'content': response})

