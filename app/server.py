from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from app.parser import DocumentParser
from app.to_db import EmbedDocuments
from app.rag_search import RagModelSearch
from pydantic import BaseModel
import httpx
import uuid
import os
import time

app = FastAPI()
parser = DocumentParser()

class RagRequest(BaseModel):
    documents: str
    questions: list[str]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/hackrx/run")
async def rag_endpoint(body: RagRequest = Body(...)):
    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            doc_url = await client.get(body.documents) # pyright: ignore[reportArgumentType]
            if doc_url.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch the PDF from URL")

            filename = f"temp_{uuid.uuid4().hex}.pdf"
            with open(filename, "wb") as f:
                f.write(doc_url.content)
        # Parse the document
        parser = DocumentParser()
        chunks = parser.parse_pdf_to_chunks(filename, source=os.path.basename(filename))
        
        embed = EmbedDocuments(chunks=chunks) # pyright: ignore[reportArgumentType]
        embed.upload_docs()
        os.remove(filename)
        
        rag = RagModelSearch()
        answers = []
        for question in body.questions:
            answer = rag.run_rag_pipeline(question)
            answers.append(answer)
        print(f"Executed in {time.time() - start} seconds.")
        return {
            "answers": answers
        }
        #return JSONdoc_url(content={"chunks": chunks})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

