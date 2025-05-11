from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

# Add the project root directory to the path to import the modules
dags_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dags_folder, "../.."))  # Adjust this path as needed
sys.path.append(project_root)
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

# Function to download Housing data from GitHub
def download_housing_data(**kwargs):
    """Download housing data from GitHub source"""
    data_dir = kwargs.get('data_dir', 'data')
    
    api = DatasetAPI(data_dir=data_dir)
    data_paths = api.download_housing_data_from_github()
    
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
    
    # Extract just the filename from the path
    input_filename = os.path.basename(input_file)
    
    # Create a symlink or copy file to raw_data_path if needed
    processor = HousingDataProcessor(config_path)
    
    # Check if file exists in raw_data_path, skip symlink creation if it does
    raw_file_path = os.path.join(processor.raw_data_path, input_filename)
    # Check if file exists in raw_data_path, remove it if it does
    if os.path.exists(raw_file_path):
        os.remove(raw_file_path)
        os.symlink(input_file, raw_file_path)
    
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
        'data_dir': '{{ var.value.data_dir | default("data/raw") }}'
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
    
    # Check if files exist before adding
    for file in \
      data/processed/cleaned_housing_{{ ds_nodash }}.csv \
      data/features/features_housing_{{ ds_nodash }}.csv \
      data/features/X_housing_{{ ds_nodash }}.csv \
      data/features/y_housing_{{ ds_nodash }}.csv
    do
      if [ -f "$file" ]; then
        dvc add "$file"
      else
        echo "Warning: File not found: $file"
      fi
    done
    
    # Push changes to remote storage
    dvc push
    
    # Only commit if there are changes
    if [ -n "$(git status --porcelain)" ]; then
      git add data/.gitignore data/processed/.gitignore data/features/.gitignore
      git commit -m "Update Housing data: {{ ds }}"
    else
      echo "No changes to commit"
    fi
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