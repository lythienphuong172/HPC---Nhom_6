FROM python:3.10-slim

WORKDIR /app

COPY dashboard_requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app_dashboard.py ./app.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
