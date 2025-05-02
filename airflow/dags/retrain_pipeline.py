from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import requests

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2023, 1, 1),  # Start date for DAG runs
}

# Function to simulate loading data into a database (should be replaced with actual DB logic)
def load_data_to_db():
    print("Simulating data load...")

# Function to trigger training via an HTTP request
def trigger_training():
    try:
        response = requests.post("http://app:8000/training")
        print(f"Training response: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error triggering training: {e}")

# Function to evaluate and compare models (MLflow or other metrics can be added here)
def evaluate_model():
    print("Comparing models...")

# Define the DAG
with DAG(
    dag_id='movie_model_retrain_pipeline',
    default_args=default_args,
    description='Retrain and evaluate movie model daily',
    schedule_interval='@daily',  # Cron expression for scheduling
    catchup=False,  # Do not backfill missed runs
    tags=['movie_model', 'retrain'],  # Optional tags for DAG categorization
) as dag:

    # Define the tasks
    start = EmptyOperator(task_id='start')  # A simple task to mark the start

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data_to_db
    )

    train_model = PythonOperator(
        task_id='trigger_training',
        python_callable=trigger_training
    )

    compare_and_register = PythonOperator(
        task_id='compare_and_register',
        python_callable=evaluate_model
    )

    # Task dependencies: the order in which tasks are executed
    start >> load_data >> train_model >> compare_and_register

