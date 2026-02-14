import os
import time
import math
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():
    docs = PyPDFLoader(str(PDF_PATH)).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                            chunk_overlap=150,
                                            add_start_index=False
                                            ).split_documents(docs)
    if not splits:
        raise SystemExit(0)
    
    enriched = [
        Document(
            page_content = d.page_content,
            metadata = {
                k: v for k, v in d.metadata.items() if v not in ("", None)
            }
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{os.getenv('GEMINI_EMBEDDING_MODEL')}")

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True
    )

    num_batches = 3
    batch_size = math.ceil(len(enriched) / num_batches)

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_docs = enriched[start:end]
        batch_ids = ids[start:end]
        if not batch_docs:
            break
        print(f"Batch {i+1}/{num_batches}: enviando {len(batch_docs)} documentos...")
        store.add_documents(batch_docs, ids=batch_ids)
        print(f"Batch {i+1}/{num_batches}: concluído.")
        if i < num_batches - 1:
            print("Aguardando 60 segundos antes do próximo batch...")
            time.sleep(60)

    print(f"Successfully ingested {len(enriched)} documents")
    
if __name__ == "__main__":
    ingest_pdf()