import streamlit as st
import tempfile
import os
import time

from rag import answer_question, ingest_documents


st.set_page_config(
    page_title="🐺 Walworth LOBO Assistant",
    layout="wide"
)

st.title("🐺 Walworth LOBO – Intelligent RAG Assistant")


# Sidebar upload
with st.sidebar:

    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        temp_paths = []

        for file in uploaded_files:

            temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            )

            temp.write(file.read())

            temp_paths.append((temp.name, file.name))

        with st.spinner("Indexing..."):

            count = ingest_documents(temp_paths)

        st.success(f"{count} chunks indexed.")


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat
for msg in st.session_state.messages:

    st.chat_message(msg["role"]).write(msg["content"])

    if msg["role"] == "assistant" and msg.get("sources"):

        with st.expander("Retrieved Sources"):

            for i, source in enumerate(msg["sources"], 1):

                st.markdown(f"### Chunk {i}")

                st.write("File:", source["metadata"].get("file_name"))
                st.write("Page:", source["metadata"].get("page"))
                st.write("Score:", source["score"])

                st.write(source["content"])


# Chat input
if question := st.chat_input("Ask a question..."):

    st.chat_message("user").write(question)

    start = time.time()

    result = answer_question(question)

    end = time.time()

    st.chat_message("assistant").write(result["answer"])

    st.caption(f"Response time: {round(end-start,2)} sec")

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
