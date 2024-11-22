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


from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Set Streamlit page configuration
st.set_page_config(page_title='🤖 RAG Chatbot with Mistral-7B-Instruct-v0.1 from HuggingfaceHub ', layout='wide')
st.title("🤖 RAG Chatbot with Mistral-7B-Instruct-v0.1 from HuggingfaceHub")

st.markdown("Q&A from private pdf documents")

# messages stores chat history for Streamlit
if "session_id" not in st.session_state:
    st.session_state.session_id = str(random.randint(1,1000))

