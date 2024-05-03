import os
import uuid
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.sensors.filesystem import FileSensor


# Correlation id for training job (this can be also found on MLFLow tracking)
correlation_id = uuid.uuid4()


def load_config():
    config_path = "ml_pipeline_config.json"  # Update the path to your JSON file
    with open(config_path) as f:
        return json.load(f)


def choose_to_do_preprocessing(**kwargs):
    config = kwargs['task_instance'].xcom_pull(task_ids='load_config_task')
    to_preprocess = config.get("enable_preprocess", False)
    if to_preprocess:
        return "preprocess_raw_audio"
    else:
        return "train_job"


# def daily_routine():
#     print("Executing daily routine...")


with DAG(
        dag_id="audiocnn_ml_pipeline_dag",
        schedule_interval=None,
        start_date=datetime(2022, 3, 3,),
        catchup=False) as dag:

    start = DummyOperator(task_id="start")

    # Task for loading the configuration from the JSON file
    load_config_task = PythonOperator(
        task_id="load_config_task",
        python_callable=load_config,
        dag=dag
    )

    # Wait for new files to arrive in the data/ directory
    wait_for_data = FileSensor(
        task_id="wait_for_data",
        filepath="data/",
        poke_interval=300,  # Check every 5 minutes
        timeout=600,  # Timeout after 10 minutes
        dag=dag
    )

    branch = BranchPythonOperator(
        task_id="check_to_preprocess_or_not",
        python_callable=choose_to_do_preprocessing,
        provide_context=True
    )

    # Task for running data preprocessing task
    preprocessing_task = BashOperator(
        task_id="preprocess_raw_audio",
        bash_command="python ${MY_LOCAL_ASSETS}/preprocess_audio.py "
                     "--dataset-version {{ dag_run.conf['preprocess']['dataset_version'] }} "
                     "--audio-dir {{ dag_run.conf['preprocess']['audio_dir'] }} "
                     "--output-dir {{ dag_run.conf['preprocess']['output_dir'] }}",
        dag=dag
    )

    script_path = "python ${MY_LOCAL_ASSETS}/train.py" + f" --correlation-id {correlation_id}"
    script_args = " --dataset-path {{ dag_run.conf['train']['dataset_path'] }} " \
                  "--n-epochs {{ dag_run.conf['train']['n_epochs'] }} " \
                  "--data-batch-size {{ dag_run.conf['train']['data_batch_size'] }} " \
                  "--model-yaml-config {{ dag_run.conf['train']['model_yaml_config'] }}"
    # Task running our ML training job
    training_task = BashOperator(
        task_id=f"train_job",
        bash_command=script_path + script_args,
        dag=dag,
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    # Task running our check for best model
    check_task = BashOperator(
        task_id='check_model',
        bash_command='python ${MY_LOCAL_ASSETS}/check_model.py',
        dag=dag,
        trigger_rule=TriggerRule.ONE_SUCCESS
    )

    complete = DummyOperator(task_id="complete", trigger_rule=TriggerRule.ONE_SUCCESS)

    # DAG which define steps to run data preprocessing and training job pipeline with input options
    start >> load_config_task >> branch
    preprocessing_task.set_upstream(branch)
    training_task.set_upstream([branch, preprocessing_task])
    check_task.set_upstream([training_task])
    complete.set_upstream(check_task)

    # Define the daily routine task
    daily_routine_task = PythonOperator(
        task_id="daily_routine",
        python_callable=daily_routine,
        dag=dag
    )

    # Schedule the daily routine task to run every day
    daily_routine_task >> start
