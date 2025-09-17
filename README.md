# Multi-Source RAG Chatbot (Streamlit + LangChain + FAISS)

Questo progetto implementa un **chatbot RAG (Retrieval-Augmented Generation)** che permette di:

- Caricare più fonti locali ( `.pdf`, `.py`, `.txt`, `.csv`, `.png`, `.jpg`)
- Estrarre e spezzare il contenuto in chunk testuali
- Creare embeddings locali con **HuggingFace**
- Salvare e ricaricare un database vettoriale **FAISS** in locale
- Recuperare i chunk rilevanti e passarli a **OpenAI GPT (via API key)**
- Fare domande in un’interfaccia **Streamlit Chat UI**

---

##  Funzionalità principali

- Supporta più tipologie di file: codice Python, PDF (testo e OCR), immagini (OCR), testo e CSV.
- Usa **FAISS** come VectorDB locale (nessun servizio esterno).
- Embeddings generati con **sentence-transformers/all-MiniLM-L6-v2** (open-source).
- **OpenAI LLM** (es. `gpt-4o-mini`, `gpt-3.5-turbo`) per generare risposte contestuali.
- Persistenza FAISS: puoi riutilizzare un DB già creato senza dover ricalcolare embeddings.
- Debug integrato: mostra quanti chunk sono stati caricati e quali documenti vengono recuperati dal retriever.

---

##  Struttura del progetto

```
rag-chatbot/
│── app.py                 # Streamlit UI
│── chatbot.py             # Chat wrapper
│── document_processor.py  # Parsing PDF, immagini, txt, csv, py
│── embedding_indexer.py   # Embeddings HuggingFace + FAISS
│── rag_chain.py           # Catena RAG moderna (prompt + retriever + LLM)
│── requirements.txt       # Dipendenze
│── .env                   # Chiave OpenAI (NON committare!)
│── README.md              # Documentazione
```

---

##  Setup

1. **Clona il repo e crea ambiente virtuale**
   ```bash
   git clone https://github.com/tuo-username/rag-chatbot.git
   cd rag-chatbot
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)
   ```

2. **Installa dipendenze**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configura la chiave OpenAI**
   Crea un file `.env` con dentro:
   ```env
   OPENAI_API_KEY=sk-xxxxxxx
   ```

---

##  Avvio dell’app

```bash
streamlit run app.py
```

Si aprirà il browser su `http://localhost:8501` con la UI.

---

## 🖼Utilizzo

1. Seleziona **🆕 Build New DB**
2. Carica uno o più file (PDF, immagini, ecc.)
3. Premi ** Run Engine** → i file vengono processati e salvati in FAISS
4. Chatta con il bot inserendo le domande nel box in basso
5. Puoi successivamente ricaricare il DB esistente scegliendo **🔄 Load Existing DB**

---

##  Debug

- In console vedrai:
  - quanti chunk sono stati aggiunti al FAISS DB
  - i primi caratteri dei documenti caricati
  - quali chunk vengono recuperati dal retriever per ogni query

Se ricevi sempre “No relevant information found”, probabilmente:
- Il file caricato non contiene testo leggibile (usa OCR fallback per immagini/PDF scansiti).
- Il prompt non riceveva correttamente il testo → già fixato in `rag_chain.py` con `RunnableLambda(format_docs)`.

---

##  Requisiti principali

- Python 3.10+  
- Streamlit  
- LangChain 0.3+  
- FAISS  
- sentence-transformers  
- OpenAI API key  

---

##  Note

- Tutto il processamento e gli embeddings avvengono **in locale**.  
- Solo la generazione finale della risposta usa **OpenAI API**.  
- Evita di committare `.env` con la tua chiave privata!  

---

Autore: *Tuo Nome*  
