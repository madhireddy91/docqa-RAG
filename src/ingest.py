from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

docs = []
for pdf in Path("data").glob("*.pdf"):
    docs.extend(PyPDFLoader(str(pdf)).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print(f"Indexed {len(chunks)} chunks from {len(docs)} pages")
