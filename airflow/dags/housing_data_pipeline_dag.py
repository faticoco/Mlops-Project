# airflow/dags/housing_data_pipeline_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

# Add the project root directory to the path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))  

from src.data.dataset_api import DatasetAPI
from src.data.housing_processor import HousingDataProcessor

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'housing_data_pipeline',
    default_args=default_args,
    description='Data processing pipeline for housing price prediction',
    schedule_interval=timedelta(days=7),  # Weekly refresh
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['ml', 'housing', 'regression'],
)

# Function to download Kaggle Housing data
def download_housing_data(**kwargs):
    """Download housing data from Kaggle/GitHub source"""
    data_dir = kwargs.get('data_dir', 'data')
    
    api = DatasetAPI(data_dir=data_dir)
    data_paths = api.download_kaggle_housing_data()
    
    # Return just the training data path
    return data_paths['train']

# Function to process housing data
def process_housing_data(**kwargs):
    """Process housing data into features for model training"""
    ti = kwargs['ti']
    config_path = kwargs['config_path']
    output_base = kwargs['output_base']
    
    # Get the input data path from XCom
    input_file = ti.xcom_pull(task_ids='download_housing_data')
    input_filename = os.path.basename(input_file)
    
    processor = HousingDataProcessor(config_path)
    result_paths = processor.process_ames_housing_pipeline(input_filename, output_base)
    
    # Return paths for XCom
    return result_paths

# Function to notify completion
def notify_completion(**kwargs):
    """Send a notification that the pipeline has completed"""
    ti = kwargs['ti']
    
    # Get the processed data paths from XCom
    result_paths = ti.xcom_pull(task_ids='process_housing_data')
    
    message = "Housing data pipeline completed successfully.\n"
    message += "Output files:\n"
    for key, path in result_paths.items():
        message += f"- {key}: {path}\n"
    
    # Here you would typically send an email or Slack notification
    # For now, we'll just print the message
    print(message)
    
    return message

# Create DAG tasks for Housing data pipeline
download_housing_data = PythonOperator(
    task_id='download_housing_data',
    python_callable=download_housing_data,
    op_kwargs={
        'data_dir': '{{ var.value.data_dir }}'
    },
    dag=dag,
)

process_housing_data = PythonOperator(
    task_id='process_housing_data',
    python_callable=process_housing_data,
    op_kwargs={
        'config_path': '{{ var.value.config_path }}/data_config.yaml',
        'output_base': 'housing_{{ ds_nodash }}.csv'
    },
    dag=dag,
)

update_dvc = BashOperator(
    task_id='update_dvc',
    bash_command='''
    cd ${AIRFLOW_HOME}/../../
    dvc add data/processed/cleaned_housing_{{ ds_nodash }}.csv
    dvc add data/features/features_housing_{{ ds_nodash }}.csv
    dvc add data/features/X_housing_{{ ds_nodash }}.csv
    dvc add data/features/y_housing_{{ ds_nodash }}.csv
    dvc push
    git add data/.gitignore data/processed/.gitignore data/features/.gitignore
    git commit -m "Update Housing data: {{ ds }}"
    ''',
    dag=dag,
)

notify_completion = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag,
)

# Define the workflow
download_housing_data >> process_housing_data >> update_dvc >> notify_completion