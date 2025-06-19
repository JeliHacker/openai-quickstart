# --- 1. Load our document and split it into chunks ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("my_document.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# --- 2. Create the Vector Store ---
# We need to turn our text chunks into numbers (embeddings) and store them.
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Use a local Ollama model for embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Use FAISS as our vector store. It will store the embeddings in memory.
# This one line does the embedding and storing for us.
vector_store = FAISS.from_documents(docs, embeddings)


# --- 3. Set up the LLM and the RAG chain ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# The prompt tells the LLM how to use the retrieved documents (the context).
prompt_template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Use the local Ollama model for generation
llm = Ollama(model="llama3")

# The retriever's job is to find the relevant documents.
retriever = vector_store.as_retriever()

# This is the RAG chain. It connects all the pieces.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --- 4. Ask a question! ---
question = "What did Dr. Thorne invent?"
print(f"Question: {question}")
print("Answer:", rag_chain.invoke(question))

print("-" * 20)

question = "What is Dr. Thorne's cat's name?"
print(f"Question: {question}")
print("Answer:", rag_chain.invoke(question))
