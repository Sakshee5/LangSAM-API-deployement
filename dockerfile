FROM python:3.12

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install git-lfs (required for Hugging Face models if used)
RUN apt-get update && apt-get install -y git-lfs

# Copy application files
COPY . .

# Set Hugging Face cache directory
ENV HF_HOME="/app/huggingface"

# Expose the port required by Hugging Face
EXPOSE 7860

# Run FastAPI with Uvicorn (on port 7860 to match Hugging Face expectations)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
