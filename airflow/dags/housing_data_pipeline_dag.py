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

# Function to download data
def download_housing_data(**kwargs):
    """Download housing data from public API"""
    dataset_type = kwargs.get('dataset_type', 'california')
    data_dir = kwargs.get('data_dir', 'data')
    
    api = DatasetAPI(data_dir=data_dir)
    
    if dataset_type == 'ames':
        return api.download_ames_housing_data()
    elif dataset_type == 'california':
        return api.fetch_boston_housing_from_sklearn()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

# Function to process housing data
def process_housing_data(**kwargs):
    """Process housing data into features for model training"""
    ti = kwargs['ti']
    dataset_type = kwargs.get('dataset_type', 'california')
    config_path = kwargs['config_path']
    output_base = kwargs['output_base']
    
    # Get the input data path from XCom
    input_file = ti.xcom_pull(task_ids=f'download_{dataset_type}_data')
    input_filename = os.path.basename(input_file)
    
    processor = HousingDataProcessor(config_path)
    
    if dataset_type == 'ames':
        result_paths = processor.process_ames_housing_pipeline(input_filename, output_base)
    elif dataset_type == 'california':
        result_paths = processor.process_california_housing_pipeline(input_filename, output_base)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Return paths for XCom
    return result_paths

# Function to notify completion
def notify_completion(**kwargs):
    """Send a notification that the pipeline has completed"""
    ti = kwargs['ti']
    dataset_type = kwargs.get('dataset_type', 'california')
    
    # Get the processed data paths from XCom
    result_paths = ti.xcom_pull(task_ids=f'process_{dataset_type}_data')
    
    message = f"Housing data pipeline completed for {dataset_type} dataset.\n"
    message += "Output files:\n"
    for key, path in result_paths.items():
        message += f"- {key}: {path}\n"
    
    # Here you would typically send an email or Slack notification
    # For now, we'll just print the message
    print(message)
    
    return message

# Create DAG tasks for California Housing dataset
download_california_data = PythonOperator(
    task_id='download_california_data',
    python_callable=download_housing_data,
    op_kwargs={
        'dataset_type': 'california',
        'data_dir': '{{ var.value.data_dir }}'
    },
    dag=dag,
)

process_california_data = PythonOperator(
    task_id='process_california_data',
    python_callable=process_housing_data,
    op_kwargs={
        'dataset_type': 'california',
        'config_path': '{{ var.value.config_path }}/data_config.yaml',
        'output_base': 'california_housing_{{ ds_nodash }}.csv'
    },
    dag=dag,
)

update_california_dvc = BashOperator(
    task_id='update_california_dvc',
    bash_command='''
    cd ${AIRFLOW_HOME}/../../
    dvc add data/processed/cleaned_california_housing_{{ ds_nodash }}.csv
    dvc add data/features/features_california_housing_{{ ds_nodash }}.csv
    dvc add data/features/X_california_housing_{{ ds_nodash }}.csv
    dvc add data/features/y_california_housing_{{ ds_nodash }}.csv
    dvc push
    git add data/.gitignore data/processed/.gitignore data/features/.gitignore
    git commit -m "Update California Housing data: {{ ds }}"
    ''',
    dag=dag,
)

notify_california_completion = PythonOperator(
    task_id='notify_california_completion',
    python_callable=notify_completion,
    op_kwargs={
        'dataset_type': 'california'
    },
    dag=dag,
)

# Create DAG tasks for Ames Housing dataset
download_ames_data = PythonOperator(
    task_id='download_ames_data',
    python_callable=download_housing_data,
    op_kwargs={
        'dataset_type': 'ames',
        'data_dir': '{{ var.value.data_dir }}'
    },
    dag=dag,
)

process_ames_data = PythonOperator(
    task_id='process_ames_data',
    python_callable=process_housing_data,
    op_kwargs={
        'dataset_type': 'ames',
        'config_path': '{{ var.value.config_path }}/data_config.yaml',
        'output_base': 'ames_housing_{{ ds_nodash }}.csv'
    },
    dag=dag,
)

update_ames_dvc = BashOperator(
    task_id='update_ames_dvc',
    bash_command='''
    cd ${AIRFLOW_HOME}/../../
    dvc add data/processed/cleaned_ames_housing_{{ ds_nodash }}.csv
    dvc add data/features/features_ames_housing_{{ ds_nodash }}.csv
    dvc add data/features/X_ames_housing_{{ ds_nodash }}.csv
    dvc add data/features/y_ames_housing_{{ ds_nodash }}.csv
    dvc push
    git add data/.gitignore data/processed/.gitignore data/features/.gitignore
    git commit -m "Update Ames Housing data: {{ ds }}"
    ''',
    dag=dag,
)

notify_ames_completion = PythonOperator(
    task_id='notify_ames_completion',
    python_callable=notify_completion,
    op_kwargs={
        'dataset_type': 'ames'
    },
    dag=dag,
)

# Define the workflow for California Housing
download_california_data >> process_california_data >> update_california_dvc >> notify_california_completion

# Define the workflow for Ames Housing
download_ames_data >> process_ames_data >> update_ames_dvc >> notify_ames_completion