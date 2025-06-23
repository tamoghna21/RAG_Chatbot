#------------------------------------------------------------------------
# Created for : Monka
# Streamlit app for RAG inference
#------------------------------------------------------------------------
import random
import streamlit as st

#from langchain_huggingface import HuggingFaceEndpoint
#from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

#from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_community.chat_message_histories import ChatMessageHistory
#from langchain_core.chat_history import BaseChatMessageHistory

#from langchain.chains import create_retrieval_chain
##from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
#
from langchain.retrievers import ContextualCompressionRetriever
from ragatouille import RAGPretrainedModel #For the Re Ranker

#from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

HUGGINGFACEHUB_API_TOKEN = st.secrets["MYHUGGINGFACEHUB_AP"]

INIT_MESSAGE = "Hi! I'm Mistral-7B-Instruct-v0.3 with RAG helper. Ask Questions."

# Set Streamlit page configuration
st.set_page_config(page_title='🤖 RAG Chatbot with Mistral-7B-Instruct-v0.2', layout='wide')
st.title("🤖 RAG Chatbot with Mistral-7B-Instruct-v0.3")

st.markdown("Q&A from private pdf documents (Federal Open Market Committee (FOMC) [meeting documents](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) for the years 2020-2023)")

