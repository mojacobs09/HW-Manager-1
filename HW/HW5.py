import streamlit as st
from openai import OpenAI
import sys
import json
import chromadb
from pathlib import Path
from pypdf import PdfReader

# fix for using chromadb on streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ── OpenAI client ──────────────────────────────────────────────────────────────
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# ── PDF helpers ────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def add_to_collection(collection, text, file_name):
    client = st.session_state.openai_client
    response = client.embeddings.create(input=text, model='text-embedding-3-small')
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
        metadatas=[{"filename": file_name}]
    )

def load_pdfs_to_collection(folder_path, collection):
    for pdf_file in Path(folder_path).glob('*.pdf'):
        text = extract_text_from_pdf(pdf_file)
        if text:
            add_to_collection(collection, text, pdf_file.name)
    return True

def create_vector_db():
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW5')
    collection = chroma_client.get_or_create_collection('HW5Collection')

    if collection.count() == 0:
        pdf_files = list(Path('./Labs/Lab-04-Data/').glob('*.pdf'))
        st.write(f"DEBUG - PDFs found: {[f.name for f in pdf_files]}")

        with st.spinner('Loading PDFs into collection...'):
            loaded = load_pdfs_to_collection('./Labs/Lab-04-Data/', collection)
            st.write(f"DEBUG - Documents in collection after load: {collection.count()}")
            st.success(f'Loaded {collection.count()} documents!')

    return collection

if 'HW5_VectorDB' not in st.session_state:
    st.session_state.HW5_VectorDB = create_vector_db()

# ── Tool function: relevant_course_info ───────────────────────────────────────
def relevant_course_info(query: str) -> str:
    client = st.session_state.openai_client

    embed_resp = client.embeddings.create(input=query, model='text-embedding-3-small')
    query_embedding = embed_resp.data[0].embedding

    results = st.session_state.HW5_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = ""
    for i in range(len(results['documents'][0])):
        doc_content = results['documents'][0][i][:2000]
        doc_name    = results['ids'][0][i]
        context += f"\n\n--- Document: {doc_name} ---\n{doc_content}\n"

    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "You are a document assistant. Given retrieved document excerpts, "
                "answer the user's question as accurately as possible. "
                "Cite the document name(s) you are drawing from. "
                "If the documents don't contain relevant info, say so clearly."
            )
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nRetrieved context:{context}"
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
            "name": "relevant_course_info",
            "description": (
                "Search the course PDF documents for information relevant to the "
                "user's question and return a synthesised answer. Use this whenever "
                "the user asks something that might be covered in the course materials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the course documents."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ── Short-term memory buffer ───────────────────────────────────────────────────
def trim_messages(messages, max_messages=6):
    system_msgs = [m for m in messages if m['role'] == 'system']
    other_msgs  = [m for m in messages if m['role'] != 'system']
    trimmed = other_msgs[-max_messages:] if len(other_msgs) > max_messages else other_msgs
    return system_msgs + trimmed

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.title('HW5: Short-Term Memory RAG Chatbot')

st.write("""
**How this chatbot works:**
- Uses **tool-calling** so the LLM itself decides when to search the course PDFs.
- The `relevant_course_info` tool embeds the LLM's query, retrieves the top-3 chunks,
  and runs a second LLM call to synthesise a document-grounded answer.
- **Short-term memory:** keeps a rolling buffer of the last 6 messages (3 exchanges).
""")

if 'hw5_messages' not in st.session_state:
    st.session_state['hw5_messages'] = [
        {
            'role': 'system',
            'content': (
                "You are a helpful assistant for students in this course. "
                "When the user asks a question that might be answered by the course materials, "
                "call the `relevant_course_info` tool with an appropriate search query. "
                "Use the tool result to ground your reply. "
                "Always let the user know which documents the information came from. "
                "After answering, ask 'Do you want more info?' and respond accordingly."
            )
        },
        {
            'role': 'assistant',
            'content': "Hi! Ask me anything about the course materials and I'll search the documents for you."
        }
    ]

for msg in st.session_state['hw5_messages']:
    if msg['role'] not in ('system', 'tool'):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

# ── Chat input & agentic loop ─────────────────────────────────────────────────
if user_input := st.chat_input('Ask a question about the course...'):
    st.session_state['hw5_messages'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    client = st.session_state.openai_client
    messages_to_send = trim_messages(st.session_state['hw5_messages'], max_messages=6)

    while True:
        response = client.chat.completions.create(
            model='gpt-5-2025-08-07',
            messages=messages_to_send,
            tools=TOOLS,
            tool_choice='auto'
        )

        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            messages_to_send.append(assistant_msg)
            st.session_state['hw5_messages'].append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                if tool_call.function.name == 'relevant_course_info':
                    args  = json.loads(tool_call.function.arguments)
                    query = args.get('query', '')

                    with st.spinner(f'Searching documents for: "{query}"...'):
                        tool_result = relevant_course_info(query)

                    tool_msg = {
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': tool_result
                    }
                    messages_to_send.append(tool_msg)
                    st.session_state['hw5_messages'].append(tool_msg)

            continue

        final_text = assistant_msg.content or ""
        with st.chat_message('assistant'):
            st.markdown(final_text)

        st.session_state['hw5_messages'].append({'role': 'assistant', 'content': final_text})
        break