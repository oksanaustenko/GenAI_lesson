from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingIndexer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

    def create_vectorstore(self, docs):
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            raise ValueError("❌ No valid documents found to embed. Check your inputs.")
        return FAISS.from_documents(docs, self.embeddings)

    def save_vectorstore(self, vectorstore, path: str = "faiss_index"):
        vectorstore.save_local(path)

    def load_vectorstore(self, path: str = "faiss_index"):
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
