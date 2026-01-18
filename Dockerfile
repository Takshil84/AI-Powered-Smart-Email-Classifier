# Dockerfile for Backend-Only Deployment (Hugging Face Spaces)
FROM python:3.9-slim
WORKDIR /app
# 1. Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 2. Copy Code & Models
# Note: We exclude the 'frontend' folder to avoid build errors
COPY app/ ./app/
COPY final_urgency_model/ ./final_urgency_model/
COPY final_bert_model/ ./final_bert_model/
COPY hybrid_inference.py .
COPY rule_based_urgency.py .
# 3. Expose Port
EXPOSE 7860
# 4. Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]