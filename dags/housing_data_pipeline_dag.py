from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys
import yaml

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import your modules
from src.data.dataset_api import DatasetAPI
from src.data.housing_processor import HousingDataProcessor
from src.models.housing_model import HousingModel  # New import for model training

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
    description='Data processing and ML pipeline for housing price prediction',
    schedule=timedelta(days=7),  # Weekly refresh
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['ml', 'housing', 'regression'],
)


# Function to download Housing data from GitHub
def download_housing_data(**kwargs):
    """Download housing data from GitHub source"""
    # Use a specific path that matches the volume mapping in docker-compose
    data_dir = '/opt/airflow/data'
    
    # Create the DatasetAPI instance with this path
    api = DatasetAPI(data_dir=data_dir)
    
    # Log the data directory
    print(f"Using data directory: {data_dir}")
    
    # Download the data
    data_paths = api.download_housing_data_from_github()
    
    # Log the paths for debugging
    print(f"Downloaded data paths: {data_paths}")
    
    # Return just the training data path
    return data_paths['train']


def process_housing_data(**kwargs):
    """Process housing data into features for model training"""
    ti = kwargs['ti']
    config_path = kwargs.get('config_path', '/opt/airflow/dags/data_config.yaml')
    output_base = kwargs.get('output_base', 'housing.csv')
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}. Creating default config.")
        config = {
            'raw_data_path': '/opt/airflow/data/raw',
            'processed_data_path': '/opt/airflow/data/processed',
            'features_path': '/opt/airflow/data/features',
            'models_path': '/opt/airflow/data/models'  # Added models directory
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Ensure all config directories exist
        for dir_path in config.values():
            os.makedirs(dir_path, exist_ok=True)
            print(f"Ensured directory exists: {dir_path}")
            
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    # Get the input data path from XCom
    input_file = ti.xcom_pull(task_ids='download_housing_data')
    print(f"Input file path: {input_file}")
    
    # Check if the file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Initialize processor
    processor = HousingDataProcessor(config_path)
    
    # Process the data using the full file path
    result_paths = processor.process_ames_housing_pipeline(input_file, output_base)
    
    # Return paths for XCom
    return result_paths


# New function for model training
def train_housing_models(**kwargs):
    """Train and evaluate machine learning models using the processed data"""
    ti = kwargs['ti']
    config_path = kwargs.get('config_path', '/opt/airflow/dags/data_config.yaml')
    
    # Get the processed data paths from XCom
    result_paths = ti.xcom_pull(task_ids='process_housing_data')
    print(f"Processing result paths: {result_paths}")
    
    # Verify we have all required paths
    if not result_paths or 'X' not in result_paths or 'y' not in result_paths or 'preprocessor' not in result_paths:
        raise ValueError("Missing required data paths from processing step")
    
    # Extract the paths to the processed data
    X_path = result_paths['X']
    y_path = result_paths['y']
    preprocessor_path = result_paths['preprocessor']
    
    # Check if files exist
    for path in [X_path, y_path, preprocessor_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Initialize the model trainer
    model_trainer = HousingModel(config_path)
    
    # Run the end-to-end training pipeline
    print("Starting model training pipeline...")
    training_results = model_trainer.end_to_end_training(
        X_path=X_path,
        y_path=y_path,
        preprocessor_path=preprocessor_path
    )
    
    print(f"Model training completed. Best model: {training_results['best_model_name']} with R² = {training_results['best_r2']:.4f}")
    
    # Return training results for next tasks
    return training_results


# Function to notify completion
def notify_completion(**kwargs):
    """Send a notification that the pipeline has completed"""
    ti = kwargs['ti']
    
    # Get the processed data paths from XCom
    processing_results = ti.xcom_pull(task_ids='process_housing_data')
    
    # Try to get training results if available
    try:
        training_results = ti.xcom_pull(task_ids='train_housing_models')
        has_training_results = True
    except:
        has_training_results = False
    
    message = "Housing data pipeline completed successfully.\n"
    
    # Report data processing results
    message += "Data Processing Results:\n"
    for key, path in processing_results.items():
        message += f"- {key}: {path}\n"
    
    # Report model training results if available
    if has_training_results:
        message += "\nModel Training Results:\n"
        message += f"- Best model: {training_results['best_model_name']}\n"
        message += f"- Best R² score: {training_results['best_r2']:.4f}\n"
        message += f"- Model saved at: {training_results['best_model_path']}\n"
    
    # Here you would typically send an email or Slack notification
    # For now, we'll just print the message
    print(message)
    
    return message


download_housing_data = PythonOperator(
    task_id='download_housing_data',
    python_callable=download_housing_data,
    dag=dag,
)

process_housing_data = PythonOperator(
    task_id='process_housing_data',
    python_callable=process_housing_data,
    op_kwargs={
        'config_path': '/opt/airflow/dags/data_config.yaml',
        'output_base': 'housing_{{ ds_nodash }}.csv'
    },
    dag=dag,
)

# New task for model training
train_housing_models = PythonOperator(
    task_id='train_housing_models',
    python_callable=train_housing_models,
    op_kwargs={
        'config_path': '/opt/airflow/dags/data_config.yaml'
    },
    dag=dag,
)

update_dvc = BashOperator(
    task_id='update_dvc',
    bash_command='''
    cd /opt/airflow
    
    # Check if files exist before adding
    for file in \
      /opt/airflow/data/processed/cleaned_housing_{{ ds_nodash }}.csv \
      /opt/airflow/data/features/features_housing_{{ ds_nodash }}.csv \
      /opt/airflow/data/features/X_housing_{{ ds_nodash }}.csv \
      /opt/airflow/data/features/y_housing_{{ ds_nodash }}.csv \
      /opt/airflow/data/models/*.pkl
    do
      if [ -f "$file" ]; then
        echo "File found: $file"
        # Uncomment if DVC is properly set up
        # dvc add "$file"
      else
        echo "Warning: File not found: $file"
      fi
    done
    ''',
    dag=dag,
)

notify_completion = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag,
)

# Define the workflow with new model training task
download_housing_data >> process_housing_data >> train_housing_models >> update_dvc >> notify_completion