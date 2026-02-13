import streamlit as st
import uuid
import traceback
import re

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------
# Load Secrets safely
# ---------------------------
def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = get_secret("AZURE_OPENAI_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT = get_secret("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBED_DEPLOYMENT = get_secret("AZURE_OPENAI_EMBED_DEPLOYMENT")
AZURE_SEARCH_ENDPOINT = get_secret("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = get_secret("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = get_secret("AZURE_SEARCH_INDEX")
OPENAI_API_VERSION = get_secret("OPENAI_API_VERSION")


# ---------------------------
# Validate config
# ---------------------------
required = [
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBED_DEPLOYMENT,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    AZURE_SEARCH_INDEX,
]

if any(v is None for v in required):
    raise Exception("Missing Azure configuration. Check Streamlit secrets.")


# ---------------------------
# DELETE INDEX (FREE TIER FIX)
# ---------------------------
def delete_index_if_exists():

    try:

        client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )

        existing_indexes = [idx.name for idx in client.list_indexes()]

        if AZURE_SEARCH_INDEX in existing_indexes:

            client.delete_index(AZURE_SEARCH_INDEX)

            print(f"Deleted existing index: {AZURE_SEARCH_INDEX}")

    except Exception as e:

        print("Index deletion skipped:", str(e))


# ---------------------------
# CLEAN PDF TEXT
# ---------------------------
def clean_pdf_text(text: str) -> str:

    if not text:
        return ""

    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ---------------------------
# CLEAN TEXT FOR PROMPT
# ---------------------------
def clean_text(text: str) -> str:

    blocked_words = [
        "ignore previous instructions",
        "system prompt",
        "assistant:",
        "user:",
        "override",
    ]

    cleaned = text

    for word in blocked_words:
        cleaned = cleaned.replace(word, "")

    cleaned = clean_pdf_text(cleaned)

    return cleaned[:1200]


# ---------------------------
# Azure OpenAI Chat
# ---------------------------
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    api_version=OPENAI_API_VERSION,
    temperature=0,
)


# ---------------------------
# Azure Embeddings
# ---------------------------
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
    api_version=OPENAI_API_VERSION,
)


# ---------------------------
# VECTOR STORE
# ---------------------------
vector_store = None


def get_vector_store():

    global vector_store

    if vector_store is None:

        vector_store = AzureSearch(
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_search_key=AZURE_SEARCH_KEY,
            index_name=AZURE_SEARCH_INDEX,
            embedding_function=embeddings.embed_query,
        )

    return vector_store


# ---------------------------
# PROMPT
# ---------------------------
prompt = ChatPromptTemplate.from_template(
"""
Answer using ONLY the provided context.

If answer is not in context, say:
I don't know.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ---------------------------
# ANSWER QUESTION
# ---------------------------
def answer_question(question: str, k: int = 3):

    try:

        store = get_vector_store()

        docs_and_scores = store.similarity_search_with_score(
            question,
            k=k
        )

        if not docs_and_scores:

            return {
                "answer": "No relevant information found in uploaded documents.",
                "sources": [],
                "no_context": True,
            }

        docs = [
            doc for doc, score in docs_and_scores
            if score < 0.6
        ]

        if not docs:

            return {
                "answer": "No relevant information found.",
                "sources": [],
                "no_context": True,
            }

        context = "\n\n".join(
            clean_text(doc.page_content)
            for doc in docs
        )

        response = llm.invoke(
            prompt.format(
                context=context,
                question=question
            )
        )

        sources = []

        for doc, score in docs_and_scores:

            sources.append({
                "content": clean_pdf_text(doc.page_content[:400]),
                "metadata": doc.metadata,
                "score": round(score, 4)
            })

        return {

            "answer": response.content,
            "sources": sources,
            "no_context": False,

        }

    except Exception:

        print(traceback.format_exc())

        return {

            "answer": "Error retrieving answer.",
            "sources": [],
            "no_context": True,

        }


# ---------------------------
# INGEST DOCUMENTS (FREE TIER SAFE)
# ---------------------------
def ingest_documents(file_paths: list[tuple[str, str]]) -> int:

    docs = []

    for temp_path, original_name in file_paths:

        loader = PyPDFLoader(temp_path)

        loaded_docs = loader.load()

        for d in loaded_docs:

            d.page_content = clean_pdf_text(d.page_content)

            d.metadata = {

                "id": str(uuid.uuid4()),
                "file_name": original_name,
                "page": d.metadata.get("page", 0) + 1,
                "source": original_name,

            }

        docs.extend(loaded_docs)

    # FREE TIER OPTIMIZED SPLITTER
    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,
        chunk_overlap=50,

    )

    chunks = splitter.split_documents(docs)

    try:

        global vector_store

        # DELETE OLD INDEX FIRST
        delete_index_if_exists()

        # CREATE NEW INDEX
        vector_store = AzureSearch.from_documents(

            documents=chunks,
            embedding=embeddings,
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_search_key=AZURE_SEARCH_KEY,
            index_name=AZURE_SEARCH_INDEX,

        )

        print(f"Ingested {len(chunks)} chunks")

    except Exception:

        print(traceback.format_exc())

        raise Exception("Azure Search index creation failed.")

    return len(chunks)
