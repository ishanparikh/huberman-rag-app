# app.py
import streamlit as st
import os
import time # To check file modification times maybe
from dotenv import load_dotenv
# LangChain components
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Free, local embeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# LLM part using Hugging Face Pipelines
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# --- Configuration ---
load_dotenv() # Loads variables from .env into environment variables

DATA_DIR = "data" # Directory where your transcript .txt files are
VECTORSTORE_DIR = "chroma_db" # Directory to save the vector database
# Use a smaller, faster embedding model suitable for CPU
# Other options: 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Use a smaller LLM suitable for CPU inference, fine-tuned for questions/answers
# Options: 'google/flan-t5-small', 'google/flan-t5-base' (larger, slower), 'MBZUAI/LaMini-Flan-T5-783M' (potentially better)
# Note: Larger models require more RAM/compute, might be very slow on CPU on free tiers
LLM_MODEL_NAME = "google/flan-t5-base" # Start with base, maybe switch to small if too slow

# Check if data directory exists and has files
if not os.path.exists(DATA_DIR) or not any(fname.endswith('.txt') for fname in os.listdir(DATA_DIR)):
    st.error(f"Error: Data directory '{DATA_DIR}' is empty or missing transcript files.")
    st.error("Please run `python load_transcripts.py` first to download transcripts.")
    st.stop() # Stop the script execution

# --- Caching functions for expensive operations ---

# Cache embedding model loading
@st.cache_resource # Use cache_resource for non-data objects like models
def get_embedding_model():
    st.write("Loading embedding model... (this may take a while on first run)")
    # Specify device='cpu' if you don't have a suitable GPU or want to force CPU
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False} # Sometimes needed
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        st.write("Embedding model loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        st.stop()


@st.cache_resource
def get_gemini_llm():
    st.write("Loading Gemini LLM...")
    try:
        # Check if API key is available
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY environment variable not set.")
            st.info("Please set the GOOGLE_API_KEY environment variable before running.")
            # You could add st.text_input here for the key ONLY for local debugging,
            # but avoid committing that code.
            st.stop()

        # Use a suitable Gemini model, e.g., gemini-1.5-flash-latest for speed/cost balance
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", # Or "gemini-1.0-pro"
                                   temperature=0.2, # Lower for more factual answers
                                   convert_system_message_to_human=True) # Helps sometimes
        st.write("Gemini LLM loaded.")
        return llm
    except Exception as e:
        st.error(f"Failed to load Gemini LLM: {e}")
        st.info("Ensure your GOOGLE_API_KEY is correct and you have internet access.")
        st.stop()


# Cache Vector Store loading/creation
@st.cache_resource(show_spinner="Setting up knowledge base (vector store)...") # Show spinner during this process
def load_or_create_vectorstore(data_dir, store_dir, _embeddings): # Pass embeddings explicitly
    # Check if the store exists and seems valid
    if os.path.exists(store_dir) and os.path.exists(os.path.join(store_dir, "chroma.sqlite3")):
        st.write(f"Loading existing vector store from {store_dir}")
        try:
            vectorstore = Chroma(persist_directory=store_dir, embedding_function=_embeddings)
            st.write("Vector store loaded.")
            return vectorstore
        except Exception as e:
             st.warning(f"Failed to load existing vector store: {e}. Will try to recreate.")
             # Consider removing the corrupted store directory here if needed
             import shutil
             shutil.rmtree(store_dir)
             os.makedirs(store_dir)

    # If store doesn't exist or failed to load, create it
    st.write(f"Creating new vector store in {store_dir}. This involves loading, splitting, and embedding documents.")
    try:
        loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader, show_progress=True, recursive=True)
        documents = loader.load()

        if not documents:
             st.error(f"No documents loaded from '{data_dir}'. Check file format and content.")
             st.stop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        texts = text_splitter.split_documents(documents)

        if not texts:
             st.error("Failed to split documents into texts. Check splitter configuration or document content.")
             st.stop()

        st.write(f"Creating embeddings for {len(texts)} text chunks... (This is the slowest part)")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=_embeddings, # Use the passed embeddings object
            persist_directory=store_dir
        )
        st.write(f"Vector store created and persisted in {store_dir}")
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        st.stop()


# --- Load Models and Vector Store ---
# These calls will use the cached resources after the first run
embeddings = get_embedding_model()
llm = get_gemini_llm() 
vectorstore = load_or_create_vectorstore(DATA_DIR, VECTORSTORE_DIR, embeddings) # Pass embeddings

# --- Setup RAG Chain ---
st.write("Setting up RAG chain...")
retriever = vectorstore.as_retriever(
    search_type="similarity", # Other options: "mmr"
    search_kwargs={"k": 3} # Retrieve top 3 relevant chunks
)

# Define the prompt template
template = """Use the following pieces of context from Huberman Lab episodes to answer the question at the end.
Provide a concise answer based *only* on the provided context. If the context doesn't contain the answer, simply state that the information wasn't found in the available transcripts.
Focus on actionable advice presented in the context if possible. Do not add information not present in the context.

Context:
{context}

Question: {question}

Answer based on context:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=False # Set to True if you want to debug sources
)
st.write("RAG chain ready.")
st.success("Application ready! Ask a question below.")


# --- Streamlit UI ---
st.title("Huberman Lab Advice Navigator")
st.caption(f"Ask about topics discussed by Andrew Huberman in the loaded transcripts. Using LLM: {LLM_MODEL_NAME}")

question = st.text_input("What topic are you interested in?", placeholder="e.g., How can I improve my sleep quality?")

if question:
    with st.spinner("Searching Huberman's advice..."):
        try:
            # Add specific query prefix if needed by model, e.g., for Flan-T5
            # query_for_llm = f"Question: {question}" # Sometimes helps guide instruction-tuned models
            result = qa_chain.invoke({"query": question}) # Use invoke for newer LangChain versions

            st.markdown("### Answer")
            st.write(result["result"])

        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")
            st.error("Please ensure the LLM and embedding models loaded correctly.")