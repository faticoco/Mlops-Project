FROM apache/airflow:3.0.0

USER root
RUN apt-get update && apt-get install -y git

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Switch back to airflow user
USER airflow