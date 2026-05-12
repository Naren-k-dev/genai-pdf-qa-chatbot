## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

Traditional chatbots are limited because they cannot access or retrieve information from custom PDF documents dynamically. Large Language Models (LLMs) alone may generate hallucinated or inaccurate responses when asked questions related to external documents.

The objective of this experiment is to develop a PDF-based Question-Answering Chatbot using LangChain and Retrieval Augmented Generation (RAG). The chatbot should be capable of loading PDF documents, extracting textual information, splitting the content into manageable chunks, converting the text into embeddings, storing the embeddings in a vector database, retrieving relevant information based on user queries, and generating accurate answers using an LLM.

### DESIGN STEPS:

#### STEP 1:

Load the PDF document using PyPDFLoader and split the extracted text into smaller chunks using RecursiveCharacterTextSplitter.

#### STEP 2:

Generate embeddings for the text chunks using OpenAIEmbeddings and store them in the Chroma vector database for semantic retrieval.

#### STEP 3:

Create a RetrievalQA chatbot using ChatOpenAI and RetrievalQA chain to answer user queries based on the retrieved PDF content.

### PROGRAM:

```
import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Load API Key
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']


# Load PDF
loader = PyPDFLoader("sample_ai_notes.pdf")

documents = loader.load()


# Split PDF into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)


# Create Embeddings
embedding = OpenAIEmbeddings()


# Create Vector Database
vectordb = Chroma.from_documents(
    docs,
    embedding
)


# Create Retriever
retriever = vectordb.as_retriever()


# Create LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)


# Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# Ask Questions
query = "What is Machine Learning?"

result = qa_chain({"query": query})

print("Name : Narendran K\nRegister Number : 212223230135")
print(result["result"])


query = "Why is Python used in AI?"

result = qa_chain({"query": query})

print(result["result"])
```

### OUTPUT:

![alt text](<EXP3 output.png>)

### RESULT:

Thus, the PDF-Based Question-Answering Chatbot was successfully developed using LangChain, OpenAI embeddings, Chroma vector database, and RetrievalQA chain. The chatbot successfully processed the PDF document, retrieved relevant information, and generated accurate answers for user queries based on the document content.

