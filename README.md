# ml-training-airflow-mlflow-example

This project is a inspired by repository https://github.com/albincorreya/ml-training-airflow-mlflow-example/'s repository. The problem with poetry is debbugged and I add some stuff of my own : a step of non-regression test.

This is an example of setting up local MLOPs training pipeline infrastructure with some dummy production ready ML research code on a local server.

This purely meant as an educational content for data scientists to get familiar with open-source tools like Airflow, MLFlow along with an example of packaing a custom python deep learning library.

The project is structured as below
- [audio_classifier](./audio_classifier): Custom python library for training deep learning audio classifiers. 
  Check [here](./audio_classifier/README.md) for more details.
- [dags](./dags): Custom Airflow DAGs and example json configurations to run ML pipeline
- [docker](./docker): Custom Dockerfiles for various containers used in our docker-compose stack.
- [env_files](./env_files): Files with environment variables defined for the docker-compose stack. (Note: dont use these secrets in production)
- [scripts](./scripts): A bunch of bash and python scripts along with some template configuration files.


## docker-compose stack


Tools used:

- Airflow: Tracking, Orchestration of DAG pipelines.
- MLFlow: Experiment tracking, versioning, model artfacts etc.
- Tensorflow, TF record, Essentia: Data processing, versioning and validation.
- Docker and compose: Containerisation all code and easy local development setup.
- Poetry: For better python dependency management of audio_classifier library.


![alt text](./assets/sketch.png)

> Note: You can also expand with stack with more custom containers with cpu or gpu support by extenting the tempalte docker-compose.yml file.

## Getting started 

- Make sure you download the dataset as per the instructions described [here](./data/README.md).

- One command to spin up everything

```
docker-compose up --build airflow
```


```
sudo chmod -R 777 mlflows
```

> Your admin password passwowrd of airflow UI can be accesed from the logs for first-time login:
Search for : Login with username: admin  password: xxxxxxxxxxxxx

### Access dashboard

- MLFlow UI: http://localhost:5000
  
- Airflow UI: http://localhost:8080

## Run DAG using Airflow

#### Using Airflow Web UI

- Go to http://localhost:8080

![alt text](./assets/dags-list.png)

You will see a list of dags we predefined defined on the [./dags]() directory.


We can also see a visualisation of ML training pipeline DAG 

![alt text](./assets/dag-example.png)


- Trigger our [ml_pipeline_dag.py](./dags/ml_pipeline_dag.py) using the 
  [ml_pipeline_config.json](./dags/ml_pipeline_config.json) example on the web UI.
  
![alt text](./assets/trigger-job-example.png)

Trigger DAG w/config  
{
    "enable_preprocess": true,
    "preprocess": {
      "dataset_version": 1,
      "audio_dir": "/opt/airflow/local-assets/raw_input_data/recordings",
      "output_dir": "/opt/airflow/local-assets/tfrecord_datasets"
    },
    "train": {
      "dataset_path": "/opt/airflow/local-assets/tfrecord_datasets/MNIST-audio/1",
      "n_epochs": 2,
      "data_batch_size": 32,
      "model_yaml_config": "/opt/airflow/dags/model-config.yaml"
    }
  }
![Example Trigger DAG w/config](./assets/trigger-job-example.png)


## Simplon

[Simplon](https://www.https://simplon.co/) is a French programming school distinguished by its inclusive approach, making programming education accessible to all. Founded on the principle of digital inclusion, Simplon offers free training in web development, AI development, and data analysis, open to everyone regardless of their starting level. Through hands-on learning and personalized support, Simplon enables individuals, including those distant from the job market, to learn digital skills and find employment in this rapidly growing sector.


## License

This project is licensed under the [MIT license](LICENSE).


## Author

This project was created by Pauline Fouquet