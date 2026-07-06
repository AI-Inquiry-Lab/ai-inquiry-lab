FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for opencv + a CJK font so matplotlib can
# render Japanese text in charts (otherwise it shows as missing-glyph boxes)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit config for Cloud Run
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["streamlit", "run", "main_app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
