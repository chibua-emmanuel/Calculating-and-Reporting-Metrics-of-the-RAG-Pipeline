import streamlit as st
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from textstat import flesch_reading_ease
import re

# Streamlit UI setup
st.title("Multi-modal Network Support RAG System - Cisco & Mikrotik")

# Placeholder for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Placeholder for RAG chain
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None

# Load the OpenAI API key from a secure location
openai_api_key = "..."  # Replace with your actual OpenAI API key

# Function to load and split PDF
def load_and_split_pdf(file_path):
    """Load a PDF file and split it into text chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to create FAISS vector store
def create_faiss_store(docs):
    """Create a FAISS vector store using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Function to initialize the RAG chain
def initialize_chain(vector_store):
    """Initialize the question-answering chain using a vector store."""
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key),
        retriever=vector_store.as_retriever()
    )
    return qa_chain

# Function to perform RAG-based QA
def perform_qa(qa_chain, user_input):
    """Perform question-answering using the RAG chain."""
    response = qa_chain({"question": user_input, "chat_history": st.session_state['chat_history']})
    answer = response.get('answer', 'No answer available')
    context = response.get('context', 'No context available')
    return answer, context

# Function to calculate retrieval metrics
def calculate_retrieval_metrics(retrieved_docs, relevant_docs):
    """Calculate precision and recall for context retrieval."""
    retrieved_set = set(retrieved_docs.split("\n"))
    relevant_set = set(relevant_docs)
    
    true_positive = len(retrieved_set.intersection(relevant_set))
    precision = true_positive / len(retrieved_docs) if retrieved_docs else 0
    recall = true_positive / len(relevant_docs) if relevant_docs else 0
    
    return precision, recall

# Function to calculate context relevance
def calculate_context_relevance(retrieved_docs, query):
    """Calculate relevance of the retrieved contexts."""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    query_embedding = embeddings.embed(query)
    context_embeddings = [embeddings.embed(doc) for doc in retrieved_docs.split("\n")]
    
    relevance_scores = [cosine_similarity(np.array(query_embedding).reshape(1, -1), 
                                          np.array(doc_embedding).reshape(1, -1))[0][0] 
                        for doc_embedding in context_embeddings]
    
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    return avg_relevance

# Function to extract entities (simple example)
def extract_entities(text):
    """Extract entities from text using a simple regex."""
    # Use regex to extract simple entities (e.g., IP addresses, device names)
    entities = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)  # Example: IP addresses
    return entities

# Function to calculate entity recall
def calculate_entity_recall(retrieved_docs, original_entities):
    """Calculate entity recall for the retrieved contexts."""
    retrieved_entities = []
    for doc in retrieved_docs.split("\n"):
        retrieved_entities.extend(extract_entities(doc))
    
    true_positive = len(set(retrieved_entities).intersection(original_entities))
    recall = true_positive / len(original_entities) if original_entities else 0
    
    return recall

# Function to calculate noise robustness
def calculate_noise_robustness(re
