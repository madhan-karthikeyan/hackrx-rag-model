import ollama
import json

# Your document
chunk = {
    "id": "2.5-i",
    "text": "Having qualified registered AYUSH Medical Practitioner in charge round the clock;",
    "metadata": {
        "section_id": "2.5",
        "clause_id": "i",
        "type": "clause",
        "source": "policy_doc",
        "title": "2.5 AYUSH Day Care Centre"
    }
}

# Run embedding using Ollama + nomic-embed-text
response = ollama.embeddings(
    model='nomic-embed-text',
    prompt=chunk["text"]
)

# Attach the embedding to your chunk
embedded_chunk = {
    "id": chunk["id"],
    "embedding": response["embedding"],
    "text": chunk["text"],
    "metadata": chunk["metadata"]
}

# Print the result
print(json.dumps(embedded_chunk, indent=2))
