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
RUN pip install --upgrade pip
RUN cat /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip list  # Debugging step to verify installed packages

# Debugging step to check if airflow is available
RUN echo $PATH  # Verify the PATH environment variable
RUN which airflow
RUN airflow version

# Initialize the Airflow database
RUN airflow db init

# Set up working directories
WORKDIR /opt/airflow

# Copy DAGs and project files
COPY airflow/dags /opt/airflow/dags