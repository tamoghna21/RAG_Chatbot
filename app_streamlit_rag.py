#------------------------------------------------------------------------
# Created by : Tamoghna Das
# Streamlit app for RAG inference
#------------------------------------------------------------------------
import random
import streamlit as st

#from langchain.prompts import PromptTemplate
#from langchain.schema.runnable import RunnablePassthrough
import os
#from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase, Result
from langchain_community.vectorstores import Neo4jVector

from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

HUGGINGFACEHUB_API_TOKEN = st.secrets["MYHUGGINGFACEHUB_AP"]

INIT_MESSAGE = "Hi! I'm Mistral-7B-Instruct-v0.1 with RAG helper. Ask Questions."

# Set Streamlit page configuration
st.set_page_config(page_title='🤖 RAG Chatbot with Mistral-7B-Instruct-v0.1', layout='wide')
st.title("🤖 RAG Chatbot with Mistral-7B-Instruct-v0.1")

st.markdown("Q&A from private pdf documents")

# messages stores chat history for Streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = str(random.randint(1,1000))

store = {}
#config = {"configurable": {"session_id": "abc2"}}
config = {"configurable": {"session_id": st.session_state.session_id}}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def init_conversationchain():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=None, #1000,
        temperature=0, #0.25,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    template = """Question: {question}

    Answer: Let's think step by step."""

    local_prompt = PromptTemplate.from_template(template)

    llm_chain = local_prompt | llm
    return llm_chain

def generate_response(conv_chain, input_text):
    #return conv_chain.invoke(input=input_text)['result']
    #return conv_chain.invoke({"input": input_text},config=config,)["answer"]
    return conv_chain.invoke({"question": input_text})

# Re-initialize the chat
def new_chat():
    st.session_state.session_id = str(random.randint(1,1000))
    st.session_state.messages = []

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# Initialize the chat
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = init_conversationchain()

# messages stores chat history for Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
with st.chat_message("assistant"):
    st.markdown(INIT_MESSAGE)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm_output = generate_response(st.session_state["rag_chain"], prompt)
        response = st.markdown(llm_output)
    
    st.session_state.messages.append({"role": "assistant", "content": llm_output})

