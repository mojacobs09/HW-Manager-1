import streamlit as st
from openai import OpenAI
import sys
import json
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup
import os

# fix for using chromadb on streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(BASE_DIR, 'HW-4-su-org')

# ── OpenAI client ──────────────────────────────────────────────────────────────
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# ── HTML helpers (from HW4) ───────────────────────────────────────────────────
def extract_text_from_html(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
    except Exception as e:
        st.error(f"Error reading HTML: {e}")
        return ""

def chunk_text(text, file_name):
    mid_point = len(text) // 2
    return [
        (f"{file_name}_chunk_1", text[:mid_point]),
        (f"{file_name}_chunk_2", text[mid_point:])
    ]

def add_to_collection(collection, text, chunk_id, file_name):
    client = st.session_state.openai_client
    response = client.embeddings.create(input=text, model='text-embedding-3-small')
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[chunk_id],
        embeddings=[embedding],
        metadatas=[{"filename": file_name, "chunk_id": chunk_id}]
    )

def load_html_to_collection(folder_path, collection):
    html_files = list(Path(folder_path).glob('*.html'))
    for html_file in html_files:
        text = extract_text_from_html(html_file)
        if text:
            chunks = chunk_text(text, html_file.name)
            for chunk_id, chunk_content in chunks:
                add_to_collection(collection, chunk_content, chunk_id, html_file.name)
    return True

def create_vector_db():
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW5')
    collection = chroma_client.get_or_create_collection('HW5Collection')
    if collection.count() == 0:
        with st.spinner('Loading HTML files into collection...'):
            loaded = load_html_to_collection(FOLDER_PATH, collection)
            st.success(f'Loaded {collection.count()} document chunks!')
    return collection

if 'HW5_VectorDB' not in st.session_state:
    st.session_state.HW5_VectorDB = create_vector_db()

# ── Tool function: relevant_club_info ─────────────────────────────────────────
def relevant_club_info(query: str) -> str:
    """
    Takes a query from the LLM, performs a vector search against the ChromaDB
    collection of student org HTML files, then invokes the LLM with the
    retrieved chunks to produce a grounded answer. Returns that answer as a
    string to be passed back as a tool result.
    """
    client = st.session_state.openai_client

    # 1. Embed the query
    embed_resp = client.embeddings.create(input=query, model='text-embedding-3-small')
    query_embedding = embed_resp.data[0].embedding

    # 2. Vector search
    results = st.session_state.HW5_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # 3. Build context from retrieved chunks
    context = ""
    for i in range(len(results['documents'][0])):
        doc_content = results['documents'][0][i][:1500]
        chunk_id    = results['ids'][0][i]
        context += f"\n\n--- Chunk: {chunk_id} ---\n{doc_content}\n"

    # 4. Invoke LLM with retrieved context (without tool-calling possibility)
    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "You are a student organization assistant. Using only the document "
                "chunks provided below, answer the user's question as accurately as "
                "possible. Cite the chunk/document name(s) you are drawing from. "
                "If the chunks don't contain relevant info, say so clearly.\n\n"
                f"Retrieved context:{context}"
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]

    synthesis_resp = client.chat.completions.create(
        model='gpt-5-2025-08-07',
        messages=synthesis_messages,
        max_tokens=600
    )

    return synthesis_resp.choices[0].message.content

# ── Tool schema for the main LLM ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the student organization HTML documents for information "
                "relevant to the user's question and return a grounded answer. "
                "Use this whenever the user asks about student clubs or organizations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the student org documents."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ── Short-term memory buffer ───────────────────────────────────────────────────
def trim_messages(messages, max_messages=10):
    """Keep system prompt + last 10 messages (5 user-assistant exchanges)"""
    system_msgs = [m for m in messages if m['role'] == 'system']
    other_msgs  = [m for m in messages if m['role'] != 'system']
    trimmed = other_msgs[-max_messages:] if len(other_msgs) > max_messages else other_msgs
    return system_msgs + trimmed

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.title('HW5: Student Organizations RAG Chatbot')

st.write("""
**How this chatbot works:**
- Uses **tool-calling** so the LLM decides when to search the student org documents.
- The `relevant_club_info` tool performs a vector search, then invokes the LLM
  with the retrieved chunks to produce a grounded, cited answer.
- **Short-term memory:** keeps a rolling buffer of the last 5 exchanges (10 messages).
""")

if 'hw5_messages' not in st.session_state:
    st.session_state['hw5_messages'] = [
        {
            'role': 'system',
            'content': (
                "You are a helpful assistant for students looking for information "
                "about student organizations. When the user asks about clubs or "
                "organizations, call the `relevant_club_info` tool with an appropriate "
                "search query. Use the tool result to ground your reply and always "
                "cite which documents the information came from."
            )
        },
        {
            'role': 'assistant',
            'content': "Hi! Ask me anything about student organizations and I'll search the documents for you."
        }
    ]

# Display history (skip system and tool messages)
for msg in st.session_state['hw5_messages']:
    if msg['role'] not in ('system', 'tool'):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'] if isinstance(msg['content'], str) else '')

# ── Chat input & agentic loop ─────────────────────────────────────────────────
if user_input := st.chat_input('Ask about student organizations...'):
    st.session_state['hw5_messages'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    client = st.session_state.openai_client
    messages_to_send = trim_messages(st.session_state['hw5_messages'], max_messages=10)

    while True:
        response = client.chat.completions.create(
            model='gpt-5-2025-08-07',
            messages=messages_to_send,
            tools=TOOLS,
            tool_choice='auto'
        )

        assistant_msg = response.choices[0].message

        # Model wants to call the tool
        if assistant_msg.tool_calls:
            messages_to_send.append(assistant_msg)
            st.session_state['hw5_messages'].append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                if tool_call.function.name == 'relevant_club_info':
                    args  = json.loads(tool_call.function.arguments)
                    query = args.get('query', '')

                    with st.spinner(f'Searching documents for: "{query}"...'):
                        tool_result = relevant_club_info(query)

                    tool_msg = {
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': tool_result
                    }
                    messages_to_send.append(tool_msg)
                    st.session_state['hw5_messages'].append(tool_msg)

            continue

        # Model produced final text reply
        final_text = assistant_msg.content or ""
        with st.chat_message('assistant'):
            st.markdown(final_text)

        st.session_state['hw5_messages'].append({'role': 'assistant', 'content': final_text})
        break