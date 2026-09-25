import os
from langchain_core.documents import Document
from vector_store import create_vector_store
from llm_chain import get_conversation_chain
from dotenv import load_dotenv

load_dotenv()

chunks = [Document(page_content="The secret password is 'Antigravity'.", metadata={"source": "dummy"})]
print("Creating vector store...")
vector_store = create_vector_store(chunks)
print("Vector store created. Creating chain...")
chain = get_conversation_chain(vector_store)
print("Chain created. Asking question...")
res = chain.invoke({'question': 'What is the secret password?'})
print("Result:", res)
