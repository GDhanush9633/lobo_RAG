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
    page_title="Walworth - LOBO AI Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Enhanced CSS Styling
# ----------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #6366f1, #8b5cf6);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #4f46e5, #7c3aed);
    }
    
    /* Chat Container */
    .chat-container {
        max-width: 900px;
        margin: auto;
        padding: 20px 0;
    }
    
    /* User Message Bubble */
    .user-msg {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 20px 20px 4px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        font-size: 16px;
        line-height: 1.6;
        animation: slideInRight 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Bot Message Bubble */
    .bot-msg {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 16px 20px;
        border-radius: 20px 20px 20px 4px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        font-size: 16px;
        line-height: 1.6;
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Typing Animation */
    .typing-indicator {
        display: inline-flex;
        gap: 4px;
        padding: 16px 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px 20px 20px 4px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .typing-indicator span {
        width: 8px;
        height: 8px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Source Chunk Card */
    .source-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #0ea5e9;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.25);
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e0e7ff !important;
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* Success Messages */
    .stSuccess {
        border-radius: 12px;
        padding: 12px 16px;
    }
    
    /* Chat Input */
    [data-testid="stChatInput"] {
        border-radius: 30px;
        border: 2px solid #e2e8f0;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #6366f1;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Stats Card */
    .stats-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .stats-card .value {
        font-size: 24px;
        font-weight: 700;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stats-card .label {
        color: #a5b4fc !important;
        font-size: 12px;
        margin-top: 4px;
    }
    
    /* Document Status */
    .doc-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        margin: 4px 0;
        font-size: 14px;
    }
    
    .doc-status .dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session State Initialization
# ----------------------------
if "typing" not in st.session_state:
    st.session_state.typing = False
if "document_stats" not in st.session_state:
    st.session_state.document_stats = {"files": 0, "chunks": 0}

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    # Logo and branding
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #e0e7ff !important; margin-bottom: 10px;">📎 Upload Documents</h2>
        <p style="color: #a5b4fc !important; font-size: 14px;">Upload your PDF documents and let LOBO answer your questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drop your PDFs here or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        with st.spinner("🔄 Indexing documents... Please wait"):
            temp_paths = []
            file_names = []
            
            for file in uploaded_files:
                file_names.append(file.name)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    temp_paths.append((tmp.name, file.name))
            
            chunks = ingest_documents(temp_paths)
            
            st.session_state.document_stats["files"] = len(uploaded_files)
            st.session_state.document_stats["chunks"] = chunks
        
        # Success message
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 16px;
            border-radius: 12px;
            margin: 16px 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        ">
            <span style="font-size: 24px;">✅</span>
            <span style="color: white; font-weight: 600;">Documents Indexed!</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Document status
        st.markdown(f"""
        <div class="doc-status">
            <span class="dot"></span>
            <span style="color: #e0e7ff;">{len(uploaded_files)} document(s) ready</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="value">{chunks}</div>
                <div class="label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="value">{len(uploaded_files)}</div>
                <div class="label">Files</div>
            </div>
            """, unsafe_allow_html=True)
        
        for temp_path, original_name in temp_paths:
            os.remove(temp_path)
    
    st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)
    
    # Retrieval Settings
    # st.markdown("### 🔍 Settings")
    # k_value = st.slider("Top-K Results", 1, 10, 5)
    
    # st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)
    
    # Actions
    st.markdown("### ⚡ Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("⬇️ Export", use_container_width=True, type="secondary"):
            chat_text = "=" * 50 + "\n"
            chat_text += "LOBO AI - Chat History\n"
            chat_text += "=" * 50 + "\n\n"
            
            for msg in st.session_state.messages:
                role = "YOU" if msg['role'] == "user" else "LOBO"
                chat_text += f"{role}: {msg['content']}\n\n"
                chat_text += "-" * 30 + "\n\n"
            
            st.download_button(
                label="Download",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )

# ----------------------------
# Initialize Chat Memory
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Welcome Banner
# ----------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px 30px;
    border-radius: 16px;
    margin-bottom: 20px;
    text-align: center;
">
    <h1 style="color: white; font-size: 28px; margin: 0;">Walworth - LOBO AI Assistant</h1>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Display Chat
# ----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        
        if msg.get("no_context"):
            st.info("💡 This question is outside the scope of uploaded documents.")
        
        elif msg.get("sources"):
            with st.expander(f"🔎 View {len(msg['sources'])} Sources", expanded=False):
                for i, source in enumerate(msg["sources"], 1):
                    file_name = source["metadata"].get("file_name", "N/A")
                    page = source["metadata"].get("page", "N/A")
                    score = source.get("score", 0.0)
                    content = source["content"].replace("\n\n", "\n").strip()
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-weight: 600; color: #0369a1;">📦 Source {i}</span>
                            <span style="
                                background: #f59e0b;
                                color: white;
                                padding: 4px 12px;
                                border-radius: 20px;
                                font-size: 12px;
                                font-weight: 600;
                            ">⭐ {score:.2f}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px;">
                            <span style="color: #64748b; font-size: 14px;">📄 {file_name}</span>
                            <span style="color: #64748b; font-size: 14px;">📑 Page {page}</span>
                        </div>
                        <div style="
                            background: white;
                            padding: 12px;
                            border-radius: 8px;
                            font-size: 14px;
                            line-height: 1.6;
                            white-space: pre-wrap;
                        ">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Chat Input
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)

# Show typing indicator
if st.session_state.typing:
    st.markdown("""
    <div class="typing-indicator">
        <span></span>
        <span></span>
        <span></span>
    </div>
    """, unsafe_allow_html=True)

question = st.chat_input("💬 Ask LOBO about your documents...")

if question:
    if len(question.strip()) < 3:
        st.warning("Please enter a more detailed question.")
        st.stop()
    
    start_time = time.time()
    st.session_state.typing = True
    
    with st.spinner("🤔 Thinking..."):
        result = answer_question(question, k=k_value)
    
    st.session_state.typing = False
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
