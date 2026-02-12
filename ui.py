import sys
from pathlib import Path
import tempfile
import os
import time
import streamlit as st
from rag import answer_question, ingest_documents

# Fix PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="LOBO AI Assistant",
    page_icon="🐺",
    layout="wide",
)

# ----------------------------
# CSS Styling
# ----------------------------
st.markdown("""
<style>
.chat-container { max-width: 900px; margin: auto; }
.user-msg {
    background-color: #2563eb;
    color: white;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.bot-msg {
    background-color: #f3f4f6;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🐺 Walworth LOBO – Intelligent RAG Assistant")
st.caption("Powered by Azure OpenAI + Azure AI Search")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📎 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Indexing documents..."):
            temp_paths = []
            file_names = []

            for file in uploaded_files:
                file_names.append(file.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    temp_paths.append((tmp.name, file.name))

                    chunks = ingest_documents(temp_paths)

            st.success("✅ Documents uploaded successfully!")
            st.info(f"📄 Files: {', '.join(file_names)}")
            st.metric("Indexed Chunks", chunks)

            for temp_path, original_name in temp_paths:
                os.remove(temp_path)


    st.divider()

    k_value = st.slider("🔎 Top-K Retrieval", 1, 5, 3)

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("⬇️ Download Chat"):
        chat_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages
        )
        st.download_button(
            label="Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

# ----------------------------
# Initialize Chat Memory
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.divider()

# ----------------------------
# Display Chat
# ----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)

    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

        if "response_time" in msg:
            st.caption(f"⏱ Response time: {msg['response_time']} sec")

        if msg.get("no_context"):
            st.info("ℹ️ This question is outside the scope of uploaded documents.")

        elif msg.get("sources"):
            with st.expander("🔎 Retrieved Sources"):

                for i, source in enumerate(msg["sources"], 1):

                    file_name = source["metadata"].get("file_name", "N/A")
                    page = source["metadata"].get("page", "N/A")
                    score = source.get("score", 0.0)

                    # Clean content formatting
                    content = source["content"].replace("\n\n", "\n").strip()

                    st.markdown(f"### 📦 Chunk {i}")

                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**📄 File:** `{file_name}`")
                        st.markdown(f"**📑 Page:** `{page}`")

                    with col2:
                        st.markdown(f"**⭐ Score:** `{score:.4f}`")

                    st.markdown(
                        f"""
                        <div style="
                            padding:12px;
                            background-color:#f9fafb;
                            border-radius:8px;
                            border:1px solid #e5e7eb;
                            font-size:14px;
                            line-height:1.6;
                            white-space: pre-wrap;
                        ">
                        {content}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.divider()


st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Chat Input
# ----------------------------
question = st.chat_input("Ask something about your documents...")

if question:

    if len(question.strip()) < 3:
        st.warning("Please enter a meaningful question.")
        st.stop()

    start_time = time.time()

    with st.spinner("Thinking..."):
        result = answer_question(question, k=k_value)

    response_time = round(time.time() - start_time, 2)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "no_context": result.get("no_context", False),
        "response_time": response_time
    })

    st.rerun()
