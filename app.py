import streamlit as st

from feedback import log_feedback
from rag import answer_question
from store import add_document, get_stats
from utils.document_processing import (
    chunk_text,
    get_file_hash,
    read_md,
    read_pdf,
    read_txt,
)

st.set_page_config(page_title="Document Q&A Assistant", layout="wide")

st.title("Document Q&A Assistant")
st.caption("Upload PDFs, TXTs, or MDs, index them, and ask grounded questions.")

# Initialize session state for feedback
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None

with st.sidebar:
    st.header("Upload & Index")
    uploaded_files = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True
    )

    chunk_size = st.slider("Chunk Size", 200, 2000, 800)
    overlap = st.slider("Overlap", 0, 500, 120)

    if st.button("Index documents"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Indexing documents..."):
                for file in uploaded_files:
                    file_hash = get_file_hash(file)

                    if file.name.endswith(".pdf"):
                        text = read_pdf(file)
                    elif file.name.endswith(".txt"):
                        text = read_txt(file)
                    elif file.name.endswith(".md"):
                        text = read_md(file)
                    else:
                        st.error(f"Unsupported file type: {file.name}")
                        continue

                    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                    added = add_document(file.name, file_hash, chunks)
                    if added:
                        st.success(f"Indexed {file.name}")
                    else:
                        st.info(f"{file.name} already indexed.")
            st.success("Indexing complete.")

    st.divider()
    st.header("Settings")
    strict_mode = st.toggle(
        "Strict Mode", value=True, help="Only answer if evidence is found in context."
    )
    min_confidence = st.slider("Min Confidence (Similarity)", 0.0, 1.0, 0.0)

    st.divider()
    st.header("Admin Stats")
    stats = get_stats()
    st.metric("Total Chunks", stats["total_chunks"])
    st.metric("Unique Documents", stats["unique_documents"])

st.subheader("Ask a question")
query = st.text_input("Enter your question about the documents:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            answer, retrieved = answer_question(
                query, strict_mode=strict_mode, min_confidence=min_confidence
            )
            st.session_state.last_query = query
            st.session_state.last_answer = answer

        st.markdown("### Answer")
        st.write(answer)

        if retrieved:
            st.markdown("### Retrieved Context")
            for i, (doc, meta, score) in enumerate(retrieved, start=1):
                with st.expander(
                    f"{i}. {meta['source']} — chunk {meta['chunk_index']} (Distance: {score:.4f})"
                ):
                    st.write(doc)
        elif strict_mode:
            st.info("No relevant context found in strict mode.")

if st.session_state.last_answer:
    st.divider()
    st.write("Was this answer helpful?")
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("👍"):
            log_feedback(
                st.session_state.last_query, st.session_state.last_answer, "thumbs_up"
            )
            st.toast("Feedback recorded!")
    with col2:
        if st.button("👎"):
            log_feedback(
                st.session_state.last_query, st.session_state.last_answer, "thumbs_down"
            )
            st.toast("Feedback recorded!")
