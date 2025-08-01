import streamlit as st
import os
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY'] # reads the system environment variable into a Python variable. 
                                          # If you only need to pass the API key to a function


## st.session_state is like a box where you can store variables that will remember their values while you use the app.
## Normally, when you refresh a Streamlit app, all variables reset.
if "vectors" not in st.session_state:
    st.session_state.loader = WebBaseLoader("https://huggingface.co/docs/transformers/index")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.embeddings = OllamaEmbeddings() # no model is passed inside because it automatically uses Ollama’s default embedding model.
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

llm = ChatGroq(groq_api_key = groq_api_key, # passing the API key to the function 
               model_name = "gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based o the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question:{input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

st.title("ChatGroq Demo")
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print("Response time :", time.process_time()-start) # (CPU time now) - (CPU time at start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"): 
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]): 
            st.write(doc.page_content)
            st.write("-------------------------------")
   

## with is just a cleaner, safer way to handle objects that need automatic opening/closing. like automatic closing of files that were opened.
## st.expander("Title") creates a collapsible box in your web app that can be expanded or collapsed by clicking it.
## enumerate() is a Python function that loops over a collection and also gives you the index (count) of each item    

## The with keyword tells Streamlit: "Whatever I write in this block should be inside the expander UI."
## The expander hides the document chunks by default (to keep the UI clean).