import streamlit as st
import uuid
import traceback
import re

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
# Clean PDF Text Properly
# ---------------------------
def clean_pdf_text(text: str) -> str:
    """
    Fix broken PDF formatting while preserving paragraphs.
    """

    if not text:
        return ""

    # Replace single newlines with space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Replace multiple newlines with paragraph break
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ---------------------------
# Clean Text for Prompt Safety
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
# Azure Search Vector Store
# ---------------------------
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=AZURE_SEARCH_INDEX,
    embedding_function=embeddings.embed_query,
)


# ---------------------------
# Prompt
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
# Answer Question
# ---------------------------
def answer_question(question: str, k: int = 3):

    try:

        docs_and_scores = vector_store.similarity_search_with_score(
            question,
            k=k
        )

        if not docs_and_scores:
            return {
                "answer": "No relevant information found in uploaded documents.",
                "sources": [],
                "no_context": True,
            }

        # docs = [doc for doc, _ in docs_and_scores]
        docs = [
    doc for doc, score in docs_and_scores
    if score < 0.05
]

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

    except Exception as e:

        print(traceback.format_exc())

        return {
            "answer": "Error retrieving answer.",
            "sources": [],
            "no_context": True,
        }


# ---------------------------
# Ingest Documents (FINAL FIXED VERSION)
# ---------------------------
def ingest_documents(file_paths: list[tuple[str, str]]) -> int:

    docs = []

    for temp_path, original_name in file_paths:

        loader = PyPDFLoader(temp_path)
        loaded_docs = loader.load()

        for d in loaded_docs:

            # Clean text BEFORE storing
            d.page_content = clean_pdf_text(d.page_content)

            # Preserve metadata
            d.metadata = {
                "id": str(uuid.uuid4()),
                "file_name": original_name,
                "page": d.metadata.get("page", 0) + 1,
                "source": original_name,
            }

        docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )

    chunks = splitter.split_documents(docs)

    try:

        # Automatically creates index with correct schema
        AzureSearch.from_documents(
            documents=chunks,
            embedding=embeddings,
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_search_key=AZURE_SEARCH_KEY,
            index_name=AZURE_SEARCH_INDEX,
        )

    except Exception as e:

        print(traceback.format_exc())
        raise Exception(f"Azure Search Upload Error: {str(e)}")

    return len(chunks)
