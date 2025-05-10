FROM apache/airflow:2.6.3-python3.9

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install additional Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN airflow db init

# Set up working directories
WORKDIR /opt/airflow

# Copy DAGs and project files
COPY airflow/dags /opt/airflow/dags