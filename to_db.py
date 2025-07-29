import json
from ollama import Client
import psycopg2
from tqdm import tqdm

# ---- Configurations ----
JSON_FILE = "chunks_output.json"

PG_CONFIG = {
    "dbname": "rag-db",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": 5432,
}

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "nomic-embed-text"

# ---- Load JSON ----
with open(JSON_FILE, "r", encoding="utf-8") as f:
    clauses = json.load(f)

# ---- Set up Ollama client ----
client = Client(host=OLLAMA_HOST)

# ---- Set up PostgreSQL ----
conn = psycopg2.connect(**PG_CONFIG)
cur = conn.cursor()

# ---- Insert each clause ----
for clause in tqdm(clauses, desc="Uploading clauses"):
    try:
        embedding_response = client.embeddings(
            model=MODEL_NAME,
            prompt=clause["text"]
        )
        embedding = embedding_response["embedding"]

        cur.execute("""
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
        conn.rollback()  # 💥 This is the fix
        print(f"❌ Failed to process clause {clause.get('id')}: {e}")


# ---- Finalize ----
conn.commit()
cur.close()
conn.close()
print("✅ All clauses processed and uploaded.")
