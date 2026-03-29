import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


from dotenv import load_dotenv
load_dotenv()


# load the GROQ api key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm_model = ChatGroq(groq_api_key = groq_api_key, model_name="Llama3-8b-8192")

prompt  = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions : {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.sessions_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_paper")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitters = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents - st.session_state.text_splitters.spit_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        



prompt = st.text_input("Enter your question from the research paper")

if st.button("document embeddings"):
    create_vector_embeddings()
    st.write("Vector database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm_model, prompt)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retriever_chains(retriever, document_chain)
    
    start = time.process_time()

    response  = retrieval_chain.invoke({"input" : user_prompt})

    print(f"Response time : {time.process_time()- start}")
    st.write(response['answer'])

    ## with a streamlit expander

    with st.expander("document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('--------------------------------')