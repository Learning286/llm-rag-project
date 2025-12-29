from langchain_community.vectorstores import FAISS
from app.embeddings import embedding_model

def get_retriever():
    db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    #return db.as_retriever(search_kwargs={"k": 3})
    return db
allow_dangerous_deserialization=True
