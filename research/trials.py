#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("ok")


# In[1]:


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[2]:


def load_pdf_file(data):

    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


# In[3]:


def text_split(extracted_data):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


# In[4]:


data = load_pdf_file("../Data/")

text_chunks = text_split(data)

print("Number of chunks:", len(text_chunks))


# In[42]:


from langchain_community.embeddings import HuggingFaceEmbeddings


# In[43]:


def download_hugging_face_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# In[44]:


embeddings = download_hugging_face_embeddings()

print(embeddings)


# In[6]:


query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))


# In[7]:


query_result


# In[45]:


from dotenv import load_dotenv
load_dotenv()


# In[46]:


from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medicalbot"

# Check if index exists before creating
if index_name not in pc.list_indexes().names():
    # Create index
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully")
else:
    print(f"Index '{index_name}' already exists")


# In[51]:


# get_ipython().run_line_magic('pip', 'install -U langchain-pinecone pinecone-client')


# In[57]:


import os
from dotenv import load_dotenv

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# In[48]:


from langchain_pinecone import Pinecone


# In[49]:


docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)


# In[52]:


from langchain_pinecone import Pinecone
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# In[53]:


docsearch


# In[54]:


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# In[55]:


retrieved_docs = retriever.invoke("What is Acne?")


# In[56]:


retrieved_docs


# In[58]:


from langchain_openai import OpenAI
llm = OpenAI(temperature=0.4, max_tokens=500)


# In[60]:


from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create the RAG chain
rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
)


# In[64]:


# Invoke the RAG chain to get the answer
try:
    response = rag_chain.invoke("What is Acne?")
    print(response)
except Exception as e:
    print(f"Error: {type(e).__name__}")
    print(f"Details: {str(e)}")
    print("\nNote: Make sure your OpenAI API key is valid and you have sufficient quota.")


# In[ ]:




