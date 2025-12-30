import mlflow
#from app.embeddings import get_embeddings_model
#from app.vectorstore import MyFAISS
#from app.model import llm_answer


# Set experiment name

from app.config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RAG-LLM-Experiment")




def log_run(query, context_docs, answer, chunk_size=500, top_k=4, temp=0.5):

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("temperature", temp)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("llm_model", "llama-3.1-8b-instant")

        # Log artifacts / results
        mlflow.log_text("\n".join(context_docs), "context_docs.txt")
        mlflow.log_text(answer, "answer.txt")
        mlflow.log_text(query, "query.txt")

        print("MLflow run logged.")
