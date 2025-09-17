import streamlit as st
from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from rag_chain import RAGChain
from chatbot import Chatbot
import os

processor = DocumentProcessor()
indexer = EmbeddingIndexer()

st.title("📚 Multi-Source RAG Chatbot with FAISS Persistence")

if "sources" not in st.session_state:
    st.session_state.sources = []

mode = st.radio("Choose mode:", ["🔄 Load Existing DB", "🆕 Build New DB"])

# === MODE: BUILD NEW DB ===
if mode == "🆕 Build New DB":
    st.subheader("Upload Sources")

    uploaded_files = st.file_uploader(
        "📂 Upload files (.py, .pdf, .png, .jpg, .csv, .txt)",
        type=["py", "pdf", "png", "jpg", "csv", "txt", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Evita duplicati
        existing_files = {src[1].name for src in st.session_state.sources if src[0] == "file"}
        for f in uploaded_files:
            if f.name not in existing_files:
                st.session_state.sources.append(("file", f))

    # Mostra le fonti correnti senza duplicati
    unique_files = {src[1].name for src in st.session_state.sources if src[0] == "file"}
    if unique_files:
        st.write("### Current Sources:")
        for fname in unique_files:
            st.write(f"- {fname}")

    if st.button("🚀 Run Engine (Build New DB)"):
        all_docs = []
        used_sources = []
        skipped_sources = []

        for _, item in st.session_state.sources:
            docs_from_source = []
            ext = item.name.split(".")[-1].lower()
            content = item.read()
            if ext == "py":
                docs_from_source = processor.from_python(content, item.name)
            elif ext == "pdf":
                docs_from_source = processor.from_pdf(content, item.name)
            elif ext in ["png", "jpg", "jpeg"]:
                docs_from_source = processor.from_image(content, item.name)
            elif ext in ["csv", "txt"]:
                text = content.decode("utf-8", errors="ignore")
                docs_from_source = processor.from_text(text, {"source": item.name, "type": ext})

            docs_from_source = [d for d in docs_from_source if d.page_content.strip()]

            if docs_from_source:
                all_docs.extend(docs_from_source)
                used_sources.append(item.name)
            else:
                skipped_sources.append(item.name)

        if not all_docs:
            st.error("❌ Nessun documento valido trovato. Controlla i file caricati.")
        else:
            vectorstore = indexer.create_vectorstore(all_docs)
            indexer.save_vectorstore(vectorstore, "faiss_index")

            rag_chain = RAGChain(vectorstore)
            st.session_state.chatbot = Chatbot(rag_chain)

            st.success("✅ New FAISS DB built and saved!")

            if used_sources:
                st.info("📂 Fonti usate:\n- " + "\n- ".join(used_sources))
            if skipped_sources:
                st.warning("⚠️ Fonti scartate (vuote o non leggibili):\n- " + "\n- ".join(skipped_sources))

# === MODE: LOAD EXISTING DB ===
elif mode == "🔄 Load Existing DB":
    if os.path.exists("faiss_index"):
        if st.button("📂 Load FAISS DB"):
            vectorstore = indexer.load_vectorstore("faiss_index")
            rag_chain = RAGChain(vectorstore)
            st.session_state.chatbot = Chatbot(rag_chain)
            st.success("✅ Existing FAISS DB loaded! Start chatting below.")
    else:
        st.warning("⚠️ No saved FAISS DB found. Please build one first.")

# === CHAT UI ===
if "chatbot" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything about your sources"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = st.session_state.chatbot.get_response(prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
