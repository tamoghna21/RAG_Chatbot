#------------------------------------------------------------------------
# Created by : Tamoghna Das
# Streamlit app for RAG inference
#------------------------------------------------------------------------
import random
import streamlit as st

from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain.retrievers import ContextualCompressionRetriever
from ragatouille import RAGPretrainedModel #For the Re Ranker

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

HUGGINGFACEHUB_API_TOKEN = st.secrets["MYHUGGINGFACEHUB_AP"]

INIT_MESSAGE = "Hi! I'm Mistral-7B-Instruct-v0.2 with RAG helper. Ask Questions."

# Set Streamlit page configuration
st.set_page_config(page_title='🤖 RAG Chatbot with Mistral-7B-Instruct-v0.2', layout='wide')
st.title("🤖 RAG Chatbot with Mistral-7B-Instruct-v0.2")

st.markdown("Q&A from private pdf documents (Federal Open Market Committee (FOMC) [meeting documents](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) for the years 2020-2023)")

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
        temperature=0.2, #0.25,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If context does not have information specific to the Question, then do not answer, say that the "
        "relevant information is not present in the retrieved context. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        #model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    db_VECTOR = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    retriever = db_VECTOR.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 10})

    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=RERANKER.as_langchain_document_compressor(), base_retriever=retriever)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain  = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="result"
    )

    #return llm_chain
    return conversational_rag_chain

def generate_response(conv_chain, input_text):
    return conv_chain.invoke({"input": input_text},config=config,)["answer"]

# Re-initialize the chat
def new_chat():
    st.session_state.session_id = str(random.randint(1,1000))
    st.session_state.messages = []

# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')
st.sidebar.write("Created by: Tamoghna Das")
st.sidebar.write("Github [link](https://github.com/tamoghna21/RAG_Chatbot/tree/main)")
st.sidebar.markdown("""
<style>
div[data-testid="stSidebarNav"]{
    flex: 1;
}
</style>
""", unsafe_allow_html=True)

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

