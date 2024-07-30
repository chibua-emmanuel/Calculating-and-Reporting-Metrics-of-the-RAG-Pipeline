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
import random

# Streamlit UI setup
st.title("Multi-modal Network Support RAG System - Cisco & Mikrotik")

# Placeholder for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Placeholder for RAG chain
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = None

# Load the OpenAI API key from a secure location
openai_api_key = "sk-proj-62mSlyzVgftvlqDEgd5nT3BlbkFJC83gxWPqhVndJfA4jrNC"  # Replace with your actual OpenAI API key

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
    context = "\n".join(st.session_state['chat_history'])  # Include past chat history as context if needed
    response = qa_chain({"question": user_input, "chat_history": st.session_state['chat_history']})
    answer = response.get('answer')
    context = response.get('context', 'No context available')
    return answer, context

# Function to calculate retrieval metrics
def calculate_retrieval_metrics(retrieved_docs, relevant_docs):
    """Calculate precision and recall for context retrieval."""
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positive = len(retrieved_set.intersection(relevant_set))
    precision = true_positive / len(retrieved_docs) if retrieved_docs else 0
    recall = true_positive / len(relevant_docs) if relevant_docs else 0
    
    return precision, recall

# Rest of your code...

# Streamlit UI to load PDF and initialize RAG system
pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(pdf_file.read())
    docs = load_and_split_pdf("uploaded_file.pdf")
    
    if docs:
        st.success("PDF Loaded and Split Successfully")
        vector_store = create_faiss_store(docs)
        st.session_state['qa_chain'] = initialize_chain(vector_store)
        st.success("RAG System Initialized")

# User interaction and metric calculations
if st.session_state['qa_chain']:
    user_input = st.text_input("Ask a question:")
    
    if user_input:
        answer, context = perform_qa(st.session_state['qa_chain'], user_input)
        st.write("Answer:", answer)
        st.write("Retrieved Context:", context)
        
        # Example relevant documents for metrics calculation
        relevant_contexts = ["Network setup guide for Cisco", "Detailed Cisco router configuration"]
        
        # Simulated ground truth and noise data for metric calculation
        ground_truths = ["The correct setup involves configuring the interfaces properly.",
                         "Security protocols are necessary for securing Cisco routers."]
        noise_contexts = ["Random noise document", "Another irrelevant document"]

        # Calculate retrieval metrics
        precision, recall = calculate_retrieval_metrics(context, relevant_contexts)
        st.write(f"Context Precision: {precision:.2f}")
        st.write(f"Context Recall: {recall:.2f}")
        
        # Calculate context relevance
        relevance_score = calculate_context_relevance(context, user_input)
        st.write(f"Context Relevance Score: {relevance_score:.2f}")
        
        # Calculate entity recall
        original_entities = ["entity1", "entity3"]
        entity_recall = calculate_entity_recall(context, original_entities)
        st.write(f"Entity Recall: {entity_recall:.2f}")
        
        # Calculate noise robustness
        noise_robustness = calculate_noise_robustness(context, noise_contexts)
        st.write(f"Noise Robustness: {noise_robustness:.2f}")
        
        # Calculate generation metrics
        faithfulness_score = calculate_faithfulness([answer], ground_truths)
        st.write(f"Faithfulness: {faithfulness_score:.2f}")

        # Calculate answer relevance
        answer_relevance = calculate_answer_relevance(answer, user_input)
        st.write(f"Answer Relevance Score: {answer_relevance:.2f}")
        
        # Calculate information integration
        information_integration = calculate_information_integration(answer)
        st.write(f"Information Integration (Readability): {information_integration:.2f}")
        
        # Calculate counterfactual robustness
        counterfactual_answers = ["A wrong setup was described.", "The answer does not align with security protocols."]
        counterfactual_robustness = calculate_counterfactual_robustness([answer], counterfactual_answers)
        st.write(f"Counterfactual Robustness: {counterfactual_robustness:.2f}")
