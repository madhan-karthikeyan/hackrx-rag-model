import httpx
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os
from ollama import Client

load_dotenv()


class RagModelSearch:
    def __init__(self):
        self.PG_CONFIG = {
            "dbname": "rag-db",
            "user": "postgres",
            "password": "password",
            "host": "localhost",
            "port": 5432,
        }
        self.client = Client(host="http://localhost:11434")
        self.model = "mistral"

    def get_query_embedding(self, text: str) -> list:
        res = httpx.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        res.raise_for_status()
        return res.json()["embedding"]

    def get_top_chunks(self, query_embedding, top_k=5):
        conn = psycopg2.connect(**self.PG_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            SELECT id, title, text, embeddings <-> (%s)::vector AS similarity
            FROM policy_clauses
            ORDER BY embeddings <-> (%s)::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))

        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    def generate_answer(self, context: str, question: str) -> str:
        prompt = f"""
You are a formal assistant answering strictly based on the provided policy text.

Context:
{context}

Question:
{question}

Instructions:
- Answer only using the provided context
- Maintain the original formal and factual tone
- Do not make assumptions or add extra commentary
- Respond in 1-2 sentences, citing facts precisely.
"""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
        )
        return response["response"].strip()

    def run_rag_pipeline(self, question: str, top_k: int = 5) -> str:
        print("Generating embedding...")
        query_embedding = self.get_query_embedding(question)

        print("Fetching top-matching clauses from DB...")
        top_chunks = self.get_top_chunks(query_embedding, top_k=top_k)

        context = "\n\n".join(f"{row[1]}: {row[2]}" for row in top_chunks)

        print("Querying GPT-3.5-turbo via aipipe...")
        answer = self.generate_answer(context, question)

        return answer
