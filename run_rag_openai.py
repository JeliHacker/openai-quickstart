import os
from dotenv import load_dotenv

# --- 0. Load environment variables from .env file ---
load_dotenv()

# Check if the API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")

# --- 1. Load our document and split it into chunks ---
# (This part is exactly the same as before)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("my_document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# --- 2. Create the Vector Store using OpenAI's embedding model ---
# This is the first key change: We use OpenAIEmbeddings instead of OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# The OpenAI embedding model will be called automatically.
embeddings = OpenAIEmbeddings()

# The rest of this is the same! FAISS works with any embedding model.
vector_store = FAISS.from_documents(docs, embeddings)


# --- 3. Set up the LLM and the RAG chain using OpenAI's model ---
from langchain_core.prompts import ChatPromptTemplate
# This is the second key change: We use ChatOpenAI instead of Ollama
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# The prompt template is the same. It's a universal instruction.
prompt_template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Here we use the ChatOpenAI model. "gpt-3.5-turbo" is fast and cheap.
llm = ChatOpenAI(model="gpt-3.5-turbo")

# The retriever is the same. It's just a tool to fetch data from the vector store.
retriever = vector_store.as_retriever()

# The RAG chain structure is identical. We just plugged in different parts.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# print(type(rag_chain))

#
# --- 4. Ask a question! ---
question = "Where does Dr. Thorne live?"
print(f"Question: {question}")
print("Answer:", rag_chain.invoke(question))

print("-" * 20)

question = "What is his cat's name?"
print(f"Question: {question}")
print("Answer:", rag_chain.invoke(question))

