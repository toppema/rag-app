from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def create_vector_store(chunks):
    """
    Creates a Chroma vector store from document chunks using Google GenAI embeddings.
    """
    # Use Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Create the vector store
    # We store it in memory for now, but you can specify a persist_directory 
    # to save it to disk for larger applications.
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings
    )
    
    return vector_store
