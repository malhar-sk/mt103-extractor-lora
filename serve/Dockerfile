FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.* "uvicorn[standard]" httpx pydantic
COPY schema.py prompts.py postprocess.py ./
COPY serve/ ./serve/
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
