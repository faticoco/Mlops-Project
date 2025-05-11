FROM apache/airflow:3.0.0

USER root
RUN apt-get update && apt-get install -y git

USER airflow
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

# Switch back to airflow user
USER airflow