import json
from ollama import Client
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv
import os

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

        self.OLLAMA_HOST = "http://localhost:11434"
        self.MODEL_NAME = "nomic-embed-text"

        # ---- Load JSON ----
        self.clauses = chunks

        # ---- Set up Ollama client ----
        self.client = Client(host=self.OLLAMA_HOST)

        # ---- Set up PostgreSQL ----
        self.conn = psycopg2.connect(**self.PG_CONFIG)
        self.cur = self.conn.cursor()

    def upload_docs(self):
        # ---- Insert each clause ----
        for clause in tqdm(self.clauses, desc="Uploading clauses"):
            try:
                embedding_response = self.client.embeddings(
                    model=self.MODEL_NAME,
                    prompt=clause["text"]
                )
                embedding = embedding_response["embedding"]

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

        # ---- Finalize ----
        self.conn.commit()
        self.cur.close()
        self.conn.close()
        print("✅ All clauses processed and uploaded.")
