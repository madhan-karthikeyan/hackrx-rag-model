from transformers import AutoTokenizer, AutoModel
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
import os
import torch
from huggingface_hub import login

load_dotenv()

class EmbedDocuments:
    def __init__(self, chunks: dict) -> None:
        self.PG_CONFIG = {
            "dbname": os.getenv("DBNAME", "rag-db"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("PASSWORD"),
            "host": os.getenv("HOST"),
            "port": os.getenv("PORT"),
        }
        self.MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.clauses = chunks

        self.conn = psycopg2.connect(**self.PG_CONFIG)
        self.cur = self.conn.cursor()

        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS policy_clauses (
                id TEXT PRIMARY KEY,
                text TEXT,
                section_id TEXT,
                clause_id TEXT,
                type TEXT,
                source TEXT,
                title TEXT,
                embeddings VECTOR(384) 
            );
        """)
        self.conn.commit()

    def upload_docs(self):
        for clause in tqdm(self.clauses, desc="Uploading clauses"):
            try:
                inputs = self.tokenizer(clause["text"], return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding_tensor = outputs.last_hidden_state.mean(dim=1)  # average pooling
                embedding = embedding_tensor.squeeze().tolist()


                self.cur.execute("""
                    INSERT INTO policy_clauses (
                        id, text, section_id, clause_id, type, source, title, embeddings
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                """, (
                    clause["id"],
                    clause["text"],
                    clause["metadata"].get("section_id"),
                    clause["metadata"].get("clause_id"),
                    clause["metadata"].get("type"),
                    clause["metadata"].get("source"),
                    clause["metadata"].get("title"),
                    embedding
                ))

            except Exception as e:
                self.conn.rollback()
                print(f"❌ Failed to process clause {clause.get('id')}: {e}")

        self.conn.commit()
        self.cur.close()
        self.conn.close()
        print("✅ All clauses processed and uploaded.")
