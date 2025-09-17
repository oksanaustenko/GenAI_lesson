class Chatbot:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def get_response(self, user_input: str):
        try:
            # Debug: cosa ha trovato il retriever
            if self.rag_chain.retriever:
                docs = self.rag_chain.retriever.get_relevant_documents(user_input)
                print(f"DEBUG: Retriever returned {len(docs)} docs")
                for i, d in enumerate(docs[:3]):  # mostra solo i primi 3
                    print(f"  Doc {i}: {d.metadata} | {d.page_content[:200]}...")

            response = self.rag_chain.chain.invoke(user_input)
            return response.content
        except Exception as e:
            return f"❌ Error: {str(e)}"
