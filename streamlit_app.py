"""
streamlit_app.py
-----------------
UI layer for "Chat With Your Own Document". All RAG logic lives in
rag_pipeline.py — this file only handles the Streamlit interface.

Run with:
    streamlit run streamlit_app.py
"""

import os
import tempfile

import streamlit as st
from rag_pipeline import build_chain, ask


# ----------------------------- Page setup ----------------------------- #
st.set_page_config(page_title="Chat With Your Own Document", )
st.title("Chat With Your Own Document")
st.caption("Upload a PDF, then ask questions about its content. Powered by RAG + Google Gemini.")


# ------------------------- Sidebar controls ---------------------------- #
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Google AI API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Get a free key (no card required) at https://aistudio.google.com/app/apikey",
    )
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    model_name = st.selectbox("Chat model", ["gemini-3.8-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"])
    st.divider()
    if st.button("Reset conversation"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("chain", None)
        st.rerun()


# ----------------------------- File upload ------------------------------ #
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and api_key:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Reading and indexing your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.session_state["chain"] = build_chain(
                pdf_path=tmp_path,
                api_key=api_key,
                model_name=model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state["file_id"] = file_id
            st.session_state["chat_history"] = []
            os.unlink(tmp_path)
        st.success(f"Indexed '{uploaded_file.name}'. Ask away!")

elif uploaded_file is not None and not api_key:
    st.warning("Enter your Google AI API key in the sidebar to continue.")


# ------------------------------ Chat UI ---------------------------------- #
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for role, message in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.markdown(message)

if "chain" in st.session_state:
    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state["chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(st.session_state["chain"], question)
                answer = result["answer"]
                sources = result["source_documents"]

                st.markdown(answer)
                if sources:
                    with st.expander(" Sources used"):
                        for i, doc in enumerate(sources, 1):
                            page = doc.metadata.get("page", "?")
                            st.markdown(f"**Chunk {i} (page {page}):**")
                            st.text(doc.page_content[:400] + "...")

        st.session_state["chat_history"].append(("assistant", answer))
else:
    st.info("Upload a PDF and enter your Google AI API key in the sidebar to start chatting.")