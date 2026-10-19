try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory

from langchain_google_genai import ChatGoogleGenerativeAI

def get_conversation_chain(vector_store):
    """
    Creates a conversational chain that uses the LLM and the vector store
    to answer questions based on the document context.
    """
    # Initialize the LLM (Gemini Flash Latest)
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    
    # Setup memory to keep track of the conversation history
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    )
    
    # Create the retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 relevant chunks
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain
