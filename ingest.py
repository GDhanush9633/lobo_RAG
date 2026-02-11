import os
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

vector_store = AzureSearch(
    azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    azure_search_key=os.environ["AZURE_SEARCH_KEY"],
    index_name=os.environ["AZURE_SEARCH_INDEX"],
    embedding_function=embeddings.embed_query,
)

loader = PyPDFLoader("data/docs/sample.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)
vector_store.add_documents(chunks)

print(f"Loaded {len(docs)} documents")
print(f"Created {len(chunks)} chunks")
print("✅ Documents embedded and indexed successfully")
