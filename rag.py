# import os
# from dotenv import load_dotenv

# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# from langchain_community.vectorstores import AzureSearch
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()

# # ---------------------------
# # Azure OpenAI (Chat)
# # ---------------------------
# llm = AzureChatOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_key=os.environ["AZURE_OPENAI_KEY"],
#     azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
#     api_version=os.environ["OPENAI_API_VERSION"],
#     temperature=0,
# )

# # ---------------------------
# # Azure OpenAI (Embeddings)
# # ---------------------------
# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_key=os.environ["AZURE_OPENAI_KEY"],
#     azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
#     api_version=os.environ["OPENAI_API_VERSION"],
# )

# # ---------------------------
# # Azure AI Search (Vector Store)
# # ---------------------------
# vector_store = AzureSearch(
#     azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
#     azure_search_key=os.environ["AZURE_SEARCH_KEY"],
#     index_name=os.environ["AZURE_SEARCH_INDEX"],
#     embedding_function=embeddings.embed_query,
# )

# # ---------------------------
# # Simple Prompt
# # ---------------------------
# prompt = ChatPromptTemplate.from_template(
# """
# Answer the question using ONLY the information provided in the context.

# If the answer is not found in the context, reply:

# I don't know.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

# # ---------------------------
# # Clean Text
# # ---------------------------
# def clean_text(text: str) -> str:
#     blocked_words = [
#         "ignore previous instructions",
#         "system prompt",
#         "act as",
#         "assistant:",
#         "user:",
#         "bypass",
#         "override",
#     ]

#     cleaned = text
#     for word in blocked_words:
#         cleaned = cleaned.replace(word, "")

#     return cleaned[:800]


# # ---------------------------
# # Answer Question
# # ---------------------------
# def answer_question(question: str, k: int = 3):

#     try:
#         docs_and_scores = vector_store.similarity_search_with_score(question, k=k)
#     except Exception as e:
#         return {
#             "answer": f"Search service error: {str(e)}",
#             "sources": [],
#             "no_context": True
#         }

#     if not docs_and_scores:
#         return {
#             "answer": "I couldn’t find relevant information in the uploaded documents.",
#             "sources": [],
#             "no_context": True
#         }

#     docs = [doc for doc, _ in docs_and_scores]

#     context = "\n\n".join(
#         clean_text(doc.page_content)
#         for doc in docs
#     )

#     try:
#         response = llm.invoke(
#             prompt.format(
#                 context=context,
#                 question=question,
#             )
#         )
#     except Exception:
#         return {
#             "answer": "Azure content filter blocked this request.",
#             "sources": [],
#             "no_context": True
#         }

#     sources = [
#         {
#             "content": doc.page_content[:300],
#             "metadata": doc.metadata,
#             "score": round(score, 4),
#         }
#         for doc, score in docs_and_scores
#     ]

#     return {
#         "answer": response.content,
#         "sources": sources,
#         "no_context": False
#     }


# # ---------------------------
# # Ingest Documents
# # ---------------------------
# def ingest_documents(file_paths: list[str]) -> int:

#     docs = []

#     for path in file_paths:
#         loader = PyPDFLoader(path)
#         loaded_docs = loader.load()

#         for d in loaded_docs:
#             d.metadata["source"] = "user_upload"
#             d.metadata["file_name"] = os.path.basename(path)

#         docs.extend(loaded_docs)

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=150,
#     )

#     chunks = splitter.split_documents(docs)
#     vector_store.add_documents(chunks)

#     return len(chunks)
# import streamlit as st

# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
# from langchain_community.vectorstores import AzureSearch
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate


# # ---------------------------
# # Load Secrets from Streamlit Cloud
# # ---------------------------
# AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
# AZURE_OPENAI_KEY = st.secrets["AZURE_OPENAI_KEY"]
# AZURE_OPENAI_CHAT_DEPLOYMENT = st.secrets["AZURE_OPENAI_CHAT_DEPLOYMENT"]
# AZURE_OPENAI_EMBED_DEPLOYMENT = st.secrets["AZURE_OPENAI_EMBED_DEPLOYMENT"]
# AZURE_SEARCH_ENDPOINT = st.secrets["AZURE_SEARCH_ENDPOINT"]
# AZURE_SEARCH_KEY = st.secrets["AZURE_SEARCH_KEY"]
# AZURE_SEARCH_INDEX = st.secrets["AZURE_SEARCH_INDEX"]
# OPENAI_API_VERSION = st.secrets["OPENAI_API_VERSION"]


# # ---------------------------
# # Azure OpenAI (Chat)
# # ---------------------------
# llm = AzureChatOpenAI(
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_KEY,
#     azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
#     api_version=OPENAI_API_VERSION,
#     temperature=0,
# )

# # ---------------------------
# # Azure OpenAI (Embeddings)
# # ---------------------------
# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_KEY,
#     azure_deployment=AZURE_OPENAI_EMBED_DEPLOYMENT,
#     api_version=OPENAI_API_VERSION,
# )

# # ---------------------------
# # Azure AI Search (Vector Store)
# # ---------------------------
# vector_store = AzureSearch(
#     azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
#     azure_search_key=AZURE_SEARCH_KEY,
#     index_name=AZURE_SEARCH_INDEX,
#     embedding_function=embeddings.embed_query,
# )

# try:
#     vector_store.client.get_index(AZURE_SEARCH_INDEX)
#     print(f"Index '{AZURE_SEARCH_INDEX}' exists.")
# except Exception:
#     print(f"Index '{AZURE_SEARCH_INDEX}' does not exist. It will be created when documents are uploaded.")

# # ---------------------------
# # Prompt
# # ---------------------------
# prompt = ChatPromptTemplate.from_template(
# """
# Answer the question using ONLY the information provided in the context.

# If the answer is not found in the context, reply:

# I don't know.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )


# # ---------------------------
# # Clean Text
# # ---------------------------
# def clean_text(text: str) -> str:
#     blocked_words = [
#         "ignore previous instructions",
#         "system prompt",
#         "act as",
#         "assistant:",
#         "user:",
#         "bypass",
#         "override",
#     ]

#     cleaned = text
#     for word in blocked_words:
#         cleaned = cleaned.replace(word, "")

#     return cleaned[:800]


# # ---------------------------
# # Answer Question
# # ---------------------------
# def answer_question(question: str, k: int = 3):

#     try:
#         docs_and_scores = vector_store.similarity_search_with_score(question, k=k)
#     except Exception as e:
#         return {
#             "answer": f"Search service error: {str(e)}",
#             "sources": [],
#             "no_context": True
#         }

#     if not docs_and_scores:
#         return {
#             "answer": "I couldn’t find relevant information in the uploaded documents.",
#             "sources": [],
#             "no_context": True
#         }

#     docs = [doc for doc, _ in docs_and_scores]

#     context = "\n\n".join(
#         clean_text(doc.page_content)
#         for doc in docs
#     )

#     try:
#         response = llm.invoke(
#             prompt.format(
#                 context=context,
#                 question=question,
#             )
#         )
#     except Exception:
#         return {
#             "answer": "Azure content filter blocked this request.",
#             "sources": [],
#             "no_context": True
#         }

#     sources = [
#         {
#             "content": doc.page_content[:300],
#             "metadata": doc.metadata,
#             "score": round(score, 4),
#         }
#         for doc, score in docs_and_scores
#     ]

#     return {
#         "answer": response.content,
#         "sources": sources,
#         "no_context": False
#     }


# # ---------------------------
# # Ingest Documents
# # ---------------------------
# def ingest_documents(file_paths: list[tuple[str, str]]) -> int:

#     docs = []

#     for path, original_name in file_paths:
#         loader = PyPDFLoader(path)
#         loaded_docs = loader.load()

#         for d in loaded_docs:
#             d.metadata["source"] = "user_upload"
#             d.metadata["file_name"] = original_name  # ✅ real file name

#         docs.extend(loaded_docs)

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=150,
#     )

#     chunks = splitter.split_documents(docs)
#     vector_store.add_documents(chunks)

#     return len(chunks)
import streamlit as st
import uuid
import traceback

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
# Clean Text
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

    return cleaned[:1000]


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
                "answer": "I don't know.",
                "sources": [],
                "no_context": True,
            }

        docs = [doc for doc, _ in docs_and_scores]

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
                "content": doc.page_content[:300],
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
# Ingest Documents (FIXED)
# ---------------------------
def ingest_documents(file_paths: list[tuple[str, str]]) -> int:

    docs = []

    try:

        for temp_path, original_name in file_paths:

            loader = PyPDFLoader(temp_path)

            loaded = loader.load()

            for d in loaded:

                d.metadata["id"] = str(uuid.uuid4())

                d.metadata["file_name"] = original_name

                d.metadata["source"] = original_name

            docs.extend(loaded)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        chunks = splitter.split_documents(docs)

        # IMPORTANT: ensure each chunk has unique ID
        for chunk in chunks:

            chunk.metadata["id"] = str(uuid.uuid4())

            if "file_name" not in chunk.metadata:
                chunk.metadata["file_name"] = "unknown.pdf"

        vector_store.add_documents(chunks)

        return len(chunks)

    except Exception as e:

        print(traceback.format_exc())

        raise Exception(f"Azure Search Upload Error: {str(e)}")

