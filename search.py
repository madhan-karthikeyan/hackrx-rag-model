import httpx
import psycopg2
import numpy as np
import dotenv

# ----- CONFIG -----
AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDE2NDZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.M8zMXIxTHMjOL9vmzn41xrEaOi1XM8rgpRY_--NmK50"  # Replace with your API key if required

PG_CONFIG = {
    "dbname": "rag-db",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": 5432,
}

QUERY = "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"

# ----- 1. Get embedding for query -----
def get_query_embedding(text: str) -> list:
    res = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    res.raise_for_status()
    return res.json()["embedding"]

# ----- 2. Retrieve top-k similar chunks -----
def get_top_chunks(query_embedding, top_k=5):
    conn = psycopg2.connect(**PG_CONFIG)
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

# ----- 3. Call GPT-3.5-Turbo via aipipe -----
def generate_answer(context, question):
    messages = [
        {
            "role": "system",
            "content": 
            """
                You are a formal assistant answering strictly based on provided policy text. 
                Respond using the same objective, third-person tone as the context. 
                Use precise, factual language with specific details, numbers, and timeframes when available.
            """
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
        AIPIPE_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.2,
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ----- MAIN -----
if __name__ == "__main__":
    print("🔎 Generating embedding...")
    query_embedding = get_query_embedding(QUERY)

    print("📦 Fetching top-matching clauses from DB...")
    top_chunks = get_top_chunks(query_embedding)

    context = "\n\n".join(f"{row[1]}: {row[2]}" for row in top_chunks)

    print("🤖 Querying GPT-3.5-turbo via aipipe...")
    answer = generate_answer(context, QUERY)

    print("\n✅ Final Answer:\n")
    print(answer)
