"""
Multi-Course AI Teaching Assistant - Streamlit Application

A production-grade RAG application enabling students to upload course documents
and have grounded, citation-backed conversations with AI.
"""
import streamlit as st
from datetime import datetime
import logging
from typing import Optional
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import CourseRetriever
from src.llm_chain import RAGChain
from src.utils import (
    sanitize_course_id,
    format_timestamp,
    get_doc_type_icon,
    get_file_icon,
    validate_course_id
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multi-Course AI Teaching Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Apple-inspired Design System
st.markdown("""
<style>
    /* ============================================
       APPLE-INSPIRED DESIGN SYSTEM
       Clean • Minimal • Seamless Transitions
       ============================================ */
    
    /* System Font Stack (Apple SF Pro fallback) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=SF+Mono:wght@400;500&display=swap');
    
    /* CSS Variables - Apple Color Palette */
    :root {
        /* Primary Colors */
        --apple-blue: #007AFF;
        --apple-blue-hover: #0056CC;
        --apple-blue-light: rgba(0, 122, 255, 0.1);
        
        /* Neutral Palette */
        --gray-50: #FAFAFA;
        --gray-100: #F5F5F7;
        --gray-200: #E8E8ED;
        --gray-300: #D2D2D7;
        --gray-400: #AEAEB2;
        --gray-500: #8E8E93;
        --gray-600: #636366;
        --gray-700: #48484A;
        --gray-800: #3A3A3C;
        --gray-900: #1D1D1F;
        
        /* Semantic Colors */
        --success: #34C759;
        --warning: #FF9500;
        --error: #FF3B30;
        
        /* Backgrounds */
        --bg-primary: #FFFFFF;
        --bg-secondary: var(--gray-100);
        --bg-tertiary: var(--gray-50);
        --bg-elevated: #FFFFFF;
        
        /* Glass Effect */
        --glass-bg: rgba(255, 255, 255, 0.72);
        --glass-border: rgba(255, 255, 255, 0.18);
        
        /* Shadows - Apple's layered shadow system */
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 30px rgba(0, 0, 0, 0.12);
        --shadow-xl: 0 20px 50px rgba(0, 0, 0, 0.15);
        
        /* Transitions - Apple's spring-like easing */
        --ease-out: cubic-bezier(0.25, 0.46, 0.45, 0.94);
        --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
        --duration-fast: 0.15s;
        --duration-normal: 0.3s;
        --duration-slow: 0.5s;
        
        /* Border Radius */
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --radius-full: 9999px;
    }
    
    /* ============================================
       GLOBAL STYLES
       ============================================ */
    
    .stApp {
        background: linear-gradient(180deg, var(--gray-100) 0%, var(--bg-primary) 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    * {
        transition: background-color var(--duration-normal) var(--ease-out),
                    border-color var(--duration-normal) var(--ease-out),
                    box-shadow var(--duration-normal) var(--ease-out),
                    transform var(--duration-fast) var(--ease-out),
                    opacity var(--duration-normal) var(--ease-out);
    }
    
    /* ============================================
       HEADER - Frosted Glass Effect
       ============================================ */
    
    .main-header {
        background: linear-gradient(135deg, var(--apple-blue) 0%, #5856D6 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2rem 2.5rem;
        border-radius: var(--radius-xl);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    }
    
    .main-header h1 {
        color: white;
        font-size: 1.875rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin: 0.625rem 0 0 0;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    /* ============================================
       CHAT MESSAGES - iMessage Style
       ============================================ */
    
    .user-message {
        background: var(--apple-blue);
        color: white;
        padding: 0.875rem 1.125rem;
        border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
        margin: 0.625rem 0;
        margin-left: 20%;
        box-shadow: var(--shadow-sm);
        font-size: 0.9375rem;
        line-height: 1.5;
        letter-spacing: -0.01em;
        animation: slideInRight var(--duration-normal) var(--ease-spring);
    }
    
    .assistant-message {
        background: var(--bg-elevated);
        color: var(--gray-900);
        padding: 0.875rem 1.125rem;
        border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
        margin: 0.625rem 0;
        margin-right: 20%;
        box-shadow: var(--shadow-md);
        font-size: 0.9375rem;
        line-height: 1.6;
        letter-spacing: -0.01em;
        border: 1px solid var(--gray-200);
        animation: slideInLeft var(--duration-normal) var(--ease-spring);
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
    
    /* ============================================
       SOURCES CARD - Subtle & Clean
       ============================================ */
    
    .sources-card {
        background: var(--gray-50);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        margin-top: 0.75rem;
        border: 1px solid var(--gray-200);
        animation: fadeIn var(--duration-normal) var(--ease-out);
    }
    
    .sources-card h4 {
        color: var(--gray-700);
        font-size: 0.8125rem;
        font-weight: 600;
        margin: 0 0 0.625rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    .source-item {
        background: var(--bg-primary);
        padding: 0.625rem 0.875rem;
        border-radius: var(--radius-sm);
        margin: 0.375rem 0;
        font-size: 0.8125rem;
        color: var(--gray-600);
        border: 1px solid var(--gray-200);
        transition: all var(--duration-fast) var(--ease-out);
    }
    
    .source-item:hover {
        background: var(--apple-blue-light);
        border-color: var(--apple-blue);
        color: var(--apple-blue);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ============================================
       EVIDENCE SNIPPETS - Code Block Style
       ============================================ */
    
    .evidence-snippet {
        background: var(--gray-900);
        color: var(--gray-100);
        padding: 1rem;
        border-radius: var(--radius-md);
        margin: 0.625rem 0;
        font-size: 0.8125rem;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        line-height: 1.6;
        border: 1px solid var(--gray-800);
        overflow-x: auto;
    }
    
    .evidence-snippet strong {
        color: var(--apple-blue);
        font-weight: 500;
    }
    
    /* ============================================
       SIDEBAR - Clean & Organized
       ============================================ */
    
    section[data-testid="stSidebar"] {
        background: var(--bg-primary);
        border-right: 1px solid var(--gray-200);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: var(--gray-900);
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: -0.01em;
    }
    
    /* ============================================
       BUTTONS - Apple Style
       ============================================ */
    
    .stButton > button {
        background: var(--apple-blue);
        color: white;
        border: none;
        border-radius: var(--radius-full);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.9375rem;
        letter-spacing: -0.01em;
        box-shadow: var(--shadow-sm);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: var(--apple-blue-hover);
        transform: scale(1.02);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Secondary Button Style */
    .stButton > button[kind="secondary"] {
        background: var(--gray-100);
        color: var(--apple-blue);
        border: 1px solid var(--gray-300);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--gray-200);
    }
    
    /* ============================================
       INPUTS - Refined & Minimal
       ============================================ */
    
    .stTextInput > div > div > input {
        border-radius: var(--radius-md);
        border: 1px solid var(--gray-300);
        padding: 0.75rem 1rem;
        font-size: 0.9375rem;
        background: var(--bg-primary);
        color: var(--gray-900);
        font-family: inherit;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--apple-blue);
        box-shadow: 0 0 0 4px var(--apple-blue-light);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: var(--gray-400);
    }
    
    .stSelectbox > div > div {
        border-radius: var(--radius-md);
        border: 1px solid var(--gray-300);
        background: var(--bg-primary);
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--gray-400);
    }
    
    /* ============================================
       CHAT INPUT - Floating Style
       ============================================ */
    
    .stChatInput {
        position: relative;
    }
    
    .stChatInput > div {
        border-radius: var(--radius-xl);
        border: 1px solid var(--gray-300);
        background: var(--bg-primary);
        box-shadow: var(--shadow-lg);
        overflow: hidden;
    }
    
    .stChatInput > div:focus-within {
        border-color: var(--apple-blue);
        box-shadow: var(--shadow-lg), 0 0 0 4px var(--apple-blue-light);
    }
    
    .stChatInput textarea {
        font-family: inherit;
        font-size: 0.9375rem;
    }
    
    /* ============================================
       METRICS / STATS
       ============================================ */
    
    [data-testid="stMetric"] {
        background: var(--bg-primary);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        border: 1px solid var(--gray-200);
        box-shadow: var(--shadow-sm);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--gray-900);
        letter-spacing: -0.02em;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8125rem;
        color: var(--gray-500);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ============================================
       EXPANDER - Collapsible Sections
       ============================================ */
    
    .streamlit-expanderHeader {
        background: var(--gray-50);
        border-radius: var(--radius-md);
        font-weight: 500;
        color: var(--gray-700);
        border: 1px solid var(--gray-200);
        padding: 0.875rem 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--gray-100);
        border-color: var(--gray-300);
    }
    
    .streamlit-expanderContent {
        border: 1px solid var(--gray-200);
        border-top: none;
        border-radius: 0 0 var(--radius-md) var(--radius-md);
        background: var(--bg-primary);
    }
    
    /* ============================================
       ALERTS & NOTIFICATIONS
       ============================================ */
    
    .stSuccess {
        background: rgba(52, 199, 89, 0.1);
        border: 1px solid rgba(52, 199, 89, 0.3);
        border-radius: var(--radius-md);
        color: #1B7D32;
    }
    
    .stWarning {
        background: rgba(255, 149, 0, 0.1);
        border: 1px solid rgba(255, 149, 0, 0.3);
        border-radius: var(--radius-md);
        color: #B86E00;
    }
    
    .stError {
        background: rgba(255, 59, 48, 0.1);
        border: 1px solid rgba(255, 59, 48, 0.3);
        border-radius: var(--radius-md);
        color: #C62828;
    }
    
    .stInfo {
        background: var(--apple-blue-light);
        border: 1px solid rgba(0, 122, 255, 0.3);
        border-radius: var(--radius-md);
        color: var(--apple-blue-hover);
    }
    
    /* ============================================
       FILE UPLOADER
       ============================================ */
    
    .stFileUploader {
        border: 2px dashed var(--gray-300);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        background: var(--gray-50);
        transition: all var(--duration-normal) var(--ease-out);
    }
    
    .stFileUploader:hover {
        border-color: var(--apple-blue);
        background: var(--apple-blue-light);
    }
    
    /* ============================================
       DIVIDERS & SPACING
       ============================================ */
    
    hr {
        border: none;
        height: 1px;
        background: var(--gray-200);
        margin: 1.5rem 0;
    }
    
    /* ============================================
       SCROLLBAR - Minimal Design
       ============================================ */
    
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gray-300);
        border-radius: var(--radius-full);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gray-400);
    }
    
    /* ============================================
       HIDE STREAMLIT BRANDING
       ============================================ */
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* ============================================
       RESPONSIVE ADJUSTMENTS
       ============================================ */
    
    @media (max-width: 768px) {
        .user-message {
            margin-left: 10%;
        }
        
        .assistant-message {
            margin-right: 10%;
        }
        
        .main-header {
            padding: 1.5rem;
        }
        
        .main-header h1 {
            font-size: 1.5rem;
        }
    }
    
    /* ============================================
       LOADING STATES
       ============================================ */
    
    .stSpinner > div {
        border-color: var(--apple-blue);
    }
    
    /* Pulse animation for loading */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = CourseRetriever(st.session_state.vector_store)
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = RAGChain(st.session_state.retriever)
    
    if "selected_course" not in st.session_state:
        st.session_state.selected_course = config.AUTO_COURSE_ID
    
    if "available_courses" not in st.session_state:
        st.session_state.available_courses = []


def refresh_courses():
    """Refresh the list of available courses."""
    st.session_state.available_courses = st.session_state.vector_store.get_all_courses()


def render_sidebar():
    """Render the sidebar with course selection and document management."""
    with st.sidebar:
        # Apple-style sidebar header
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <h2 style="font-size: 1.25rem; font-weight: 600; color: #1D1D1F; margin: 0; letter-spacing: -0.02em;">
                📚 Course Assistant
            </h2>
            <p style="font-size: 0.8125rem; color: #8E8E93; margin: 0.25rem 0 0 0;">
                AI-powered study companion
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Course selector
        st.markdown("#### Course Selection")
        
        refresh_courses()
        course_options = [config.AUTO_COURSE_ID] + st.session_state.available_courses + [config.MISC_COURSE_ID]
        course_options = list(dict.fromkeys(course_options))  # Remove duplicates while preserving order
        
        selected = st.selectbox(
            "Active Course",
            options=course_options,
            index=course_options.index(st.session_state.selected_course) if st.session_state.selected_course in course_options else 0,
            help="Select a course to filter questions, or use AUTO to let the system detect the course."
        )
        st.session_state.selected_course = selected
        
        if selected == config.AUTO_COURSE_ID:
            st.info("🔍 AUTO mode: The system will attempt to detect the relevant course from your question.")
        
        st.markdown("---")
        
        # Upload section
        st.markdown("#### 📤 Upload Documents")
        
        with st.container():
            # Course ID for upload
            upload_course_id = st.text_input(
                "Course ID",
                placeholder="e.g., CS101, MATH201",
                help="Enter the course ID for these documents"
            )
            
            # Document type selector
            doc_type = st.selectbox(
                "Document Type",
                options=config.DOCUMENT_TYPES,
                format_func=lambda x: f"{get_doc_type_icon(x)} {x.title()}"
            )
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload Files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload PDF or TXT files"
            )
            
            # Upload button
            if st.button("📥 Upload & Index", use_container_width=True, type="primary"):
                if not upload_course_id:
                    st.error("Please enter a Course ID")
                elif not uploaded_files:
                    st.error("Please select files to upload")
                else:
                    is_valid, error_msg = validate_course_id(upload_course_id)
                    if not is_valid:
                        st.error(error_msg)
                    else:
                        with st.spinner("Processing documents..."):
                            process_uploads(uploaded_files, upload_course_id, doc_type)
        
        st.markdown("---")
        
        # Document list for selected course
        if selected and selected != config.AUTO_COURSE_ID:
            st.markdown(f"#### 📁 Documents in {selected}")
            
            docs = st.session_state.vector_store.get_documents_by_course(selected)
            
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        icon = get_file_icon(doc['file_type'])
                        type_icon = get_doc_type_icon(doc['doc_type'])
                        st.markdown(f"""
                        <div style="font-size: 0.85rem; margin-bottom: 0.5rem;">
                            {icon} <strong>{doc['source_file']}</strong><br>
                            <span style="color: #64748b; font-size: 0.75rem;">
                                {type_icon} {doc['doc_type'].title()} • {doc['total_pages']} page(s)
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['source_file']}", help="Delete document"):
                            if st.session_state.vector_store.delete_document(selected, doc['source_file']):
                                st.success(f"Deleted {doc['source_file']}")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
            else:
                st.info("No documents uploaded for this course yet.")
        
        st.markdown("---")
        
        # Stats
        stats = st.session_state.vector_store.get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats['total_chunks'])
        with col2:
            st.metric("Courses", len(stats['courses']))
        
        # Reset conversation
        st.markdown("---")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def process_uploads(uploaded_files, course_id: str, doc_type: str):
    """Process uploaded files and add to vector store."""
    sanitized_course = sanitize_course_id(course_id)
    
    files_data = []
    for file in uploaded_files:
        files_data.append({
            'content': file.read(),
            'name': file.name
        })
    
    documents, errors = st.session_state.doc_processor.process_multiple_files(
        files=files_data,
        course_id=sanitized_course,
        doc_type=doc_type
    )
    
    if documents:
        ids = st.session_state.vector_store.add_documents(documents)
        st.success(f"✅ Successfully indexed {len(documents)} chunks from {len(files_data) - len(errors)} file(s)")
        refresh_courses()
        
        # Update selected course to the newly uploaded one
        if st.session_state.selected_course == config.AUTO_COURSE_ID:
            st.session_state.selected_course = sanitized_course
    
    for error in errors:
        st.error(error)


def render_chat_message(role: str, content: str, sources: Optional[list] = None, evidence: Optional[list] = None):
    """Render a chat message with optional sources and evidence."""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Render sources
        if sources:
            sources_html = "".join([
                f'<div class="source-item">📄 {s["source_file"]} (Page {s["page_number"]}) - {s["doc_type"].title()}</div>'
                for s in sources
            ])
            st.markdown(f"""
            <div class="sources-card">
                <h4>📚 Sources ({len(sources)})</h4>
                {sources_html}
            </div>
            """, unsafe_allow_html=True)
        
        # Render evidence in expander
        if evidence:
            with st.expander(f"📋 View Evidence ({len(evidence)} snippets)"):
                for i, e in enumerate(evidence, 1):
                    meta = e['metadata']
                    st.markdown(f"""
                    <div class="evidence-snippet">
                        <strong>Source {i}:</strong> {meta.get('source_file', 'Unknown')} (Page {meta.get('page_number', 'N/A')})<br><br>
                        {e['content'][:500]}{'...' if len(e['content']) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)


def render_main_chat():
    """Render the main chat interface."""
    # Apple-style Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Multi-Course AI Teaching Assistant</h1>
        <p>Ask anything about your course materials. Get instant, cited answers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("""
        ⚠️ **OpenAI API Key Required**
        
        Please set your `OPENAI_API_KEY` environment variable or add it to a `.env` file in the project root.
        
        ```
        OPENAI_API_KEY=your-api-key-here
        ```
        """)
        return
    
    # Display current course context
    if st.session_state.selected_course != config.AUTO_COURSE_ID:
        st.info(f"📚 Currently filtering by: **{st.session_state.selected_course}**")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(
                role=msg["role"],
                content=msg["content"],
                sources=msg.get("sources"),
                evidence=msg.get("evidence")
            )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your course materials..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Render user message immediately
        render_chat_message("user", prompt)
        
        # Check if there are any documents
        stats = st.session_state.vector_store.get_collection_stats()
        if stats['total_chunks'] == 0:
            no_docs_msg = "📚 No documents have been uploaded yet. Please upload some course materials using the sidebar to get started."
            st.session_state.messages.append({
                "role": "assistant",
                "content": no_docs_msg
            })
            render_chat_message("assistant", no_docs_msg)
            st.rerun()
            return
        
        # Generate response with streaming
        with st.spinner("Searching documents and generating response..."):
            try:
                # Prepare chat history for context
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]  # Exclude current message
                ]
                
                # Get course ID for filtering
                course_id = st.session_state.selected_course
                
                # Generate streaming response
                response_placeholder = st.empty()
                full_response = ""
                sources = None
                evidence = None
                
                for chunk_data in st.session_state.rag_chain.generate_response_stream(
                    question=prompt,
                    course_id=course_id,
                    chat_history=chat_history
                ):
                    if chunk_data["sources"] is not None:
                        sources = chunk_data["sources"]
                        evidence = chunk_data["evidence"]
                    
                    full_response += chunk_data["chunk"]
                    response_placeholder.markdown(f"""
                    <div class="assistant-message">
                        {full_response}▌
                    </div>
                    """, unsafe_allow_html=True)
                
                # Final render without cursor
                response_placeholder.empty()
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "evidence": evidence
                })
                
                # Render final message with sources
                render_chat_message("assistant", full_response, sources, evidence)
                
            except Exception as e:
                error_msg = f"⚠️ Error generating response: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
        
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_chat()


if __name__ == "__main__":
    main()

