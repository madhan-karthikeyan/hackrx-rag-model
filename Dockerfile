FROM python:3.11-slim

WORKDIR /app

COPY . .

# System dependencies (curl, libgl1 for fitz/PyMuPDF, build-essential for psycopg2)
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]