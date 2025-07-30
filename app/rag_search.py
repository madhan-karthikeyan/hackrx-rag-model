import httpx
import psycopg2
import numpy as np
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch

load_dotenv()


class RagModelSearch:
    def __init__(self):
        self.AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"
        self.API_KEY = os.getenv("AIPIPE_KEY")
        self.PG_CONFIG = {
            "dbname": "rag-db",
            "user": "postgres",
            "password": "password",
            "host": "localhost",
            "port": 5432,
        }
        self.embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        self.embed_model = AutoModel.from_pretrained(self.embed_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_model.to(self.device)

    def get_query_embedding(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
        return embedding

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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a formal assistant answering strictly based on provided policy text. "
                    "Respond using the same objective, third-person tone as the context. "
                    "Use precise, factual language with specific details, numbers, and timeframes when available."
                )
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question: {question}

Instructions:
- Answer only using the provided context
- Maintain the original formal and factual tone
- Do not make assumptions or add extra commentary
- Respond in 1-2 sentences, citing facts precisely
"""
            }
        ]

        response = httpx.post(
            self.AIPIPE_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.API_KEY}"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.15,
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def run_rag_pipeline(self, question: str, top_k: int = 5) -> str:
        print("Generating embedding...")
        query_embedding = self.get_query_embedding(question)

        print("Fetching top-matching clauses from DB...")
        top_chunks = self.get_top_chunks(query_embedding, top_k=top_k)

        context = "\n\n".join(f"{row[1]}: {row[2]}" for row in top_chunks)

        print(f"Querying GPT-3.5-turbo via aipipe...")
        answer = self.generate_answer(context, question)

        return answer

