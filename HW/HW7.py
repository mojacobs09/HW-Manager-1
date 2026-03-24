import streamlit as st
from openai import OpenAI
import sys
import json
import chromadb
import pandas as pd

# fix for chromadb on streamlit
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ── OpenAI client ──────────────────────────────────────────────────────────────
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

client = st.session_state.openai_client

# ── ChromaDB setup ─────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path='./news_chromadb')
collection    = chroma_client.get_or_create_collection('news')

# ── Load CSV into ChromaDB (only runs once when collection is empty) ───────────
uploaded = st.sidebar.file_uploader('Upload news.csv', type='csv')

if uploaded and collection.count() == 0:
    df = pd.read_csv(uploaded)
    with st.spinner('Loading articles into ChromaDB...'):
        for i, row in df.iterrows():
            text = str(row.get('Document', '')).strip()
            if not text:
                continue
            embedding = client.embeddings.create(input=text, model='text-embedding-3-small').data[0].embedding
            collection.add(
                documents=[text],
                ids=[f'article_{i}'],
                embeddings=[embedding],
                metadatas=[{
                    'source': str(row.get('company_name', '')),
                    'date':   str(row.get('Date', ''))[:10],
                    'url':    str(row.get('URL', '')),
                }]
            )
    st.sidebar.success(f'Loaded {collection.count()} articles!')

# ── Model selector ─────────────────────────────────────────────────────────────
model = st.sidebar.selectbox('Model', ['gpt-4o-mini', 'gpt-4o'])

# ── Tool functions ─────────────────────────────────────────────────────────────
def search_articles(query, top_k=6):
    embedding = client.embeddings.create(input=query, model='text-embedding-3-small').data[0].embedding
    results   = collection.query(query_embeddings=[embedding], n_results=top_k)
    context   = ''
    for i in range(len(results['documents'][0])):
        meta     = results['metadatas'][0][i]
        context += f"\n[{i+1}] {meta['source']} | {meta['date']}\n{results['documents'][0][i][:500]}\nURL: {meta['url']}\n"
    return context

def get_interesting_news(top_k=8):
    # broad query to surface varied important stories
    embedding = client.embeddings.create(input='major business news financial update announcement', model='text-embedding-3-small').data[0].embedding
    results   = collection.query(query_embeddings=[embedding], n_results=top_k * 3)
    # diversity filter: max 2 per company
    seen, filtered = {}, []
    for i in range(len(results['documents'][0])):
        meta   = results['metadatas'][0][i]
        source = meta['source']
        if seen.get(source, 0) < 2:
            filtered.append((results['documents'][0][i], meta))
            seen[source] = seen.get(source, 0) + 1
        if len(filtered) >= top_k:
            break
    context = ''
    for i, (doc, meta) in enumerate(filtered, 1):
        context += f"\n[{i}] {meta['source']} | {meta['date']}\n{doc[:500]}\nURL: {meta['url']}\n"
    return context

# ── Tools schema ───────────────────────────────────────────────────────────────
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_articles',
            'description': 'Search news articles by topic or company. Use for specific queries like "find news about JPMorgan".',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query'},
                    'top_k': {'type': 'integer', 'description': 'Number of results (default 6)'}
                },
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_interesting_news',
            'description': 'Return the most interesting and newsworthy stories. Use when the user asks for top or interesting news.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'top_k': {'type': 'integer', 'description': 'Number of stories (default 8)'}
                }
            }
        }
    }
]

SYSTEM = ('You are a financial news analyst. Always call a tool before answering. '
          'For top/interesting news call get_interesting_news and explain why each story matters. '
          'For specific topics call search_articles. Return a numbered ranked list, cite company, date, and URL.')

# ── Short-term memory ──────────────────────────────────────────────────────────
def trim_messages(messages, max_messages=10):
    system_msgs = [m for m in messages if m['role'] == 'system']
    other_msgs  = [m for m in messages if m['role'] != 'system']
    return system_msgs + other_msgs[-max_messages:]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title('📰 News Reporting Bot')

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'system',    'content': SYSTEM},
        {'role': 'assistant', 'content': 'Hi! Upload your news.csv in the sidebar and ask me anything.\n- *Find the most interesting news*\n- *Find news about JPMorgan*'}
    ]

for msg in st.session_state.messages:
    if msg['role'] not in ('system', 'tool'):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'] if isinstance(msg['content'], str) else '')

if user_input := st.chat_input('Ask about the news...'):
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    messages_to_send = trim_messages(st.session_state.messages)

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages_to_send,
            tools=TOOLS,
            tool_choice='auto'
        )
        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            messages_to_send.append(assistant_msg)
            st.session_state.messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                with st.spinner(f'Searching: "{args.get("query", "top news")}"...'):
                    if tool_call.function.name == 'search_articles':
                        result = search_articles(args.get('query', ''), args.get('top_k', 6))
                    elif tool_call.function.name == 'get_interesting_news':
                        result = get_interesting_news(args.get('top_k', 8))
                    else:
                        result = 'Unknown tool.'

                tool_msg = {'role': 'tool', 'tool_call_id': tool_call.id, 'content': result}
                messages_to_send.append(tool_msg)
                st.session_state.messages.append(tool_msg)
            continue

        final_text = assistant_msg.content or ''
        with st.chat_message('assistant'):
            st.markdown(final_text)
        st.session_state.messages.append({'role': 'assistant', 'content': final_text})
        break