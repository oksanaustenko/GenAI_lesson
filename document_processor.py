import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz  # PyMuPDF per PDF
import pytesseract
from PIL import Image

class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def from_text(self, text: str, metadata=None):
        text = text.strip()
        if not text:
            return []
        doc = [Document(page_content=text, metadata=metadata or {})]
        return self.splitter.split_documents(doc)

    def from_python(self, file_content: bytes, filename: str):
        text = file_content.decode("utf-8", errors="ignore")
        return self.from_text(text, {"source": filename, "type": "python"})

    def from_pdf(self, file_content: bytes, filename: str):
        docs = []
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            tmp.flush()
            doc = fitz.open(tmp.name)
            for page in doc:
                text = page.get_text("text")
                if not text.strip():  # fallback OCR
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                docs.extend(self.from_text(text, {"source": filename, "type": "pdf"}))
        return docs

    def from_image(self, file_content: bytes, filename: str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(file_content)
            tmp.flush()
            img = Image.open(tmp.name)
            text = pytesseract.image_to_string(img)
        return self.from_text(text, {"source": filename, "type": "image"})
