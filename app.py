import streamlit as st
import os
from dotenv import load_dotenv

from document_processor import process_document
from vector_store import create_vector_store
from llm_chain import get_conversation_chain

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

def handle_user_input(user_question):
    """
    Passes the user's question to the conversation chain and displays the response.
    """
    if st.session_state.conversation:
        # Append user message to UI state
        st.session_state.messages.append({
            "role": "user",
            "content": user_question,
            "sources": []
        })
        with st.chat_message("user"):
            st.write(user_question)
            
        # Fetch and display the assistant's answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({'question': user_question})
                    answer_content = response.get('answer', '')
                    sources = response.get('source_documents', [])
                except Exception as e:
                    answer_content = f"⚠️ Could not generate an answer: {e}. Please try asking again."
                    sources = []
                
            st.write(answer_content)
                    
            # Display source citations for this AI response
            if sources:
                with st.expander("📚 Click here to see the raw text pulled from the document"):
                    for idx, doc in enumerate(sources):
                        st.info(f"**Chunk {idx + 1} from your document:**\n\n{doc.page_content[:300]}...")
                        
        # Append assistant response to UI state so it persists across reruns
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_content,
            "sources": sources
        })
        if st.session_state.conversation and hasattr(st.session_state.conversation, 'memory'):
            st.session_state.chat_history = st.session_state.conversation.memory.buffer
    else:
        st.warning("Please upload and process a document first.")

def main():
    # Page configuration for a professional look
    st.set_page_config(page_title="DocuMind - AI Document Analyst", page_icon="📚", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
    .doc-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 12px;
        border: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_doc_name" not in st.session_state:
        st.session_state.current_doc_name = None

    @st.cache_resource(show_spinner=False)
    def load_demo_vector_store():
        with open("resume.pdf", "rb") as f:
            chunks = process_document(f)
        return create_vector_store(chunks)

    # Main UI Header
    st.title("📚 DocuMind AI")
    st.markdown("### Your Intelligent Document Assistant")
    st.markdown("Upload a PDF document and ask questions about its content. Powered by **Google Gemini** and **LangChain**.")
    
    st.divider()

    # Sidebar for uploading files
    with st.sidebar:
        # API Key status indicator
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.warning("⚠️ GOOGLE_API_KEY is not set.")
            user_key = st.text_input("Enter Google API Key", type="password", help="Enter your Gemini API key to activate the app.")
            if user_key:
                os.environ["GOOGLE_API_KEY"] = user_key.strip()
                st.success("API Key set!")
                st.rerun()
        else:
            st.caption("🔑 API Key: **Connected** ✅")

        st.divider()
        st.subheader("Your Documents")
        uploaded_file = st.file_uploader(
            "Upload a Document", 
            type=["pdf", "docx", "html", "htm"], 
            help="Maximum file size is limited to 5MB to ensure fast processing."
        )
        
        # Notice if user uploaded a different file than currently active
        if uploaded_file is not None and st.session_state.current_doc_name and uploaded_file.name != st.session_state.current_doc_name:
            st.info(f"🔄 New file selected: **{uploaded_file.name}**. Click **Process Document** to switch contexts.")
        
        if st.button("Process Document", type="primary", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("Analyzing document..."):
                    chunks = process_document(uploaded_file)
                    vector_store = create_vector_store(chunks)
                    # Initialize fresh conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    # Reset conversation history for the new document
                    st.session_state.chat_history = None
                    st.session_state.messages = []
                    st.session_state.current_doc_name = uploaded_file.name
                st.success(f"Processed '{uploaded_file.name}'! Chat history reset.")
                st.rerun()
            else:
                st.error("Please upload a document first.")
                
        st.divider()
        st.subheader("Try it out!")
        if st.button("Try Demo (My Resume)", use_container_width=True):
            if os.path.exists("resume.pdf"):
                with st.spinner("Analyzing resume (from cache)..."):
                    vector_store = load_demo_vector_store()
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.session_state.chat_history = None
                    st.session_state.messages = []
                    st.session_state.current_doc_name = "resume.pdf (Demo)"
                st.success("Demo Resume processed! Chat history reset.")
                st.rerun()
            else:
                st.error("No 'resume.pdf' found in the folder. Please add it first!")

        st.divider()
        # Reset / Clear Chat Button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = None
                if st.session_state.conversation and hasattr(st.session_state.conversation, 'memory'):
                    st.session_state.conversation.memory.clear()
                st.rerun()

        st.markdown("""
        **About this app:**
        This is an end-to-end RAG (Retrieval-Augmented Generation) application built to demonstrate advanced NLP and Vector Search capabilities.
        """)

    # Main Chat Interface Header with Active Document Indicator and Clear Button
    col_hdr, col_clear = st.columns([3, 1])
    with col_hdr:
        st.subheader("Chat Interface")
    with col_clear:
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", key="clear_chat_top"):
                st.session_state.messages = []
                st.session_state.chat_history = None
                if st.session_state.conversation and hasattr(st.session_state.conversation, 'memory'):
                    st.session_state.conversation.memory.clear()
                st.rerun()

    # Active Document status badge
    if st.session_state.current_doc_name:
        st.markdown(f"📄 **Active Document:** `{st.session_state.current_doc_name}`")
    else:
        st.info("👆 Please upload and click **Process Document** in the sidebar to start asking questions.")
    
    # Display persistent chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Click here to see the raw text pulled from the document"):
                    for idx, doc in enumerate(msg["sources"]):
                        st.info(f"**Chunk {idx + 1} from your document:**\n\n{doc.page_content[:300]}...")

    # Accept new user input
    user_question = st.chat_input("Ask a question about your document...")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()
