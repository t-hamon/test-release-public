from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import requests

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

def load_data_to_db():
    # Ideally this should load fresh data into PostgreSQL
    print("Simulating data load...")

def trigger_training():
    response = requests.post("http://app:8000/training")
    print("Training response:", response.status_code, response.text)

def evaluate_model():
    # Use mlflow APIs to fetch old & new metrics, and compare
    print("Comparing models...")

with DAG(
    dag_id='movie_model_retrain_pipeline',
    default_args=default_args,
    description='Retrain and evaluate movie model daily',
    schedule_interval='@daily',  # Simulated daily trigger
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    start = DummyOperator(task_id='start')

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

    start >> load_data >> train_model >> compare_and_register

