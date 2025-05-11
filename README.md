# Innovate Analytics Inc. - ML System

This repository contains the code and infrastructure for the machine learning system developed for Innovate Analytics Inc.

## Project Structure

- `data/`: Contains raw and processed data
- `src/`: Contains source code for the project
- `airflow/`: Contains Airflow DAGs and operators
- `notebooks/`: Contains Jupyter notebooks for exploration
- `tests/`: Contains unit and integration tests
- `config/`: Contains configuration files
- `docs/`: Contains documentation
- `mlflow/`: MLflow tracking directory
- `models/`: Contains trained models

## Setup

1. Clone this repository
2. Create and activate virtual environment
3. Install dependencies from requirements.txt
4. docker build -t airflow-dvc .
5. docker images

## Team

- Member 1: Data Engineering and Model Development
- Member 2: CI/CD with Jenkins and Docker
- Member 3: Infrastructure and Platform Engineering
  raw_data_path: 'data/raw/'
  processed_data_path: 'data/processed/'
  features_path: 'data/features/'
