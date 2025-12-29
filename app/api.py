from fastapi import FastAPI, UploadFile
from app.ingestion import load_and_split
from app.embeddings import create_vector_store
from app.retrieval import get_retriever
from app.llm import generate_answer
from experiments.mlflow_tracking import log_run
app = FastAPI()

@app.get("/")
def home():
    return {"msg": "RAG running!"}

@app.post("/upload")
def upload_pdf(file: UploadFile):
    path = f"data/documents/{file.filename}"

    with open(path, "wb") as f:
        f.write(file.file.read())

    chunks = load_and_split(path)
    create_vector_store(chunks)

    return {"status": "Document indexed"}

@app.post("/query")
def query_llm(query: str):
    retriever = get_retriever()
    #docs = retriever.get_relevant_documents(query)
    docs=retriever.similarity_search(query, k=5)
   
    context = "\n".join([d.page_content for d in docs])

    answer = generate_answer(context, query)

    #return {"answer": answer}
    log_run(query=query, context_docs=[d.page_content for d in docs], answer=answer, chunk_size=500, top_k=4, temp=0.5)

    return {
        "query": query,
        "context_used": [d.page_content for d in docs],
        "answer": answer
    }




