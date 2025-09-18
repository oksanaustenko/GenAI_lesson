from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()
        self.retriever = None
        self.chain = self.create_chain()

    def get_llm(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY in .env")

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key
        )

    def create_chain(self):
        self.retriever = self.vectorstore.as_retriever(search_type='mmr',
                                                       search_kwargs={"k": 10,"fetch_k":50,
                                                                      "lambda_mult":0.2})

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        template = """You are an assistant that answers questions using the context provided.
            Answer as precisely as possible using only the context. 
            If you truly cannot find the answer, don't say: 
            'I could not find relevant info, try to evaluate your answer on documents you have retrieved'
        
            Context:
            {context}
        
            Question: {question}
        
            Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        return (
                {"context": self.retriever | RunnableLambda(format_docs),
                 "question": RunnablePassthrough()}
                | prompt
                | self.llm
        )

