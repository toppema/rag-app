import streamlit as st
import os
from dotenv import load_dotenv

from document_processor import process_document
from vector_store import create_vector_store
from llm_chain import get_conversation_chain

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

def handle_user_input(user_question):
    """
    Passes the user's question to the conversation chain and displays the response.
    """
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)
    else:
        st.warning("Please upload and process a document first.")

def main():
    # Page configuration for a professional look
    st.set_page_config(page_title="DocuMind - AI Document Analyst", page_icon="📚", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main UI Header
    st.title("📚 DocuMind AI")
    st.markdown("### Your Intelligent Document Assistant")
    st.markdown("Upload a PDF document and ask questions about its content. Powered by **Google Gemini** and **LangChain**.")
    
    st.divider()

    # Sidebar for uploading files
    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        
        if st.button("Process Document"):
            if uploaded_file is not None:
                with st.spinner("Analyzing document..."):
                    # 1. Process and chunk the document
                    chunks = process_document(uploaded_file)
                    
                    # 2. Create the vector store
                    vector_store = create_vector_store(chunks)
                    
                    # 3. Create the conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    
                st.success("Document processed successfully! You can now ask questions.")
            else:
                st.error("Please upload a PDF document first.")
                
        st.divider()
        st.markdown("""
        **About this app:**
        This is an end-to-end RAG (Retrieval-Augmented Generation) application built to demonstrate advanced NLP and Vector Search capabilities.
        """)

    # Main Chat Interface
    st.subheader("Chat Interface")
    user_question = st.chat_input("Ask a question about your document...")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    # Ensure the user has provided an API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY is not set. Please set it in your environment variables or a .env file.")
        st.stop()
        
    main()
